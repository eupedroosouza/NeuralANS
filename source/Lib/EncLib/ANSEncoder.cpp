#include "ANSEncoder.h"

#include <cmath>

ANSEncoder::ANSEncoder(const Context &context) : context(context) {
    // Initialize states with first states.
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 13; j++) {
            contextualizedStates[i][j] = this->context.getContext(j, static_cast<TensorType>(i))->getFirstState();
        }
    }
}

void ANSEncoder::encodeBin(const uint8_t bin, const uint8_t ctxId, const TensorType paramType) {
    const Table *table = context.getContext(ctxId, paramType);
    const auto pType = static_cast<uint8_t>(paramType);
    table->encode(this->contextualizedStates[pType][ctxId], static_cast<int8_t>(bin), writer);
}

void ANSEncoder::encodeBinEP(const uint8_t bin) {
    writer.write(1, bin);
}

void ANSEncoder::encodeBinsEP(const uint32_t bins, const uint32_t numBins) {
    if (numBins < 32)
        CHECK(bins >= ( 1u << numBins ), printf( "%i can not be coded with %i EP-Bins\n", bins, numBins ))
    writer.write(numBins, bins);
}

void ANSEncoder::iae_v(const uint8_t bitwidth, const int32_t value) {
    const uint32_t pattern = static_cast<uint32_t>(value) & (0xFFFFFFFF >> (32 - bitwidth));
    encodeBinsEP(pattern, bitwidth);
}

void ANSEncoder::uae_v(const uint8_t bitwidth, const uint32_t value) {
    encodeBinsEP(value, bitwidth);
}



void ANSEncoder::encodeWeightBAC(int32_t value, uint8_t k) {
    const uint32_t sigFlag = value != 0 ? 1 : 0;
    const int32_t sigCtx = contextModeler.getSigCtxId();

    if (sigFlag) {
        const uint32_t signFlag = value < 0 ? 1 : 0;
        const int32_t signCtx = contextModeler.getSignFlagCtxId();

        uint32_t remAbsLevel = abs(value) - 1;
        if (abs(value) > 5) {
            remAbsLevel -= 5;
            encodeAbsRem(remAbsLevel, k);
            encodeBin(1, 12, tensorType);
        } else {
            uint32_t grXFlag = remAbsLevel ? 1 : 0;
            const uint32_t staticGrXFlag = grXFlag;
            int32_t ctxId = contextModeler.getGtxCtxId(signFlag);
            const int32_t staticCtxId = ctxId;
            uint32_t numGreaterFlagsCoded = 1;
            std::pair<uint32_t, int32_t> enc[3];
            int encPtr = 0;
            while (grXFlag && (numGreaterFlagsCoded < 4)) // TODO: 4? (based only StaticBAC)
            {
                remAbsLevel--;
                grXFlag = remAbsLevel ? 1 : 0;
                ctxId = contextModeler.getGtxCtxId(signFlag);
                enc[encPtr++] = {grXFlag, ctxId};
                numGreaterFlagsCoded++;
            }
            for (int i = encPtr - 1; i >= 0; i--) {
                const auto [e_grXFlag, e_ctxId] = enc[i];
                encodeBin(e_grXFlag, e_ctxId, tensorType);
            }
            encodeBin(staticGrXFlag, staticCtxId, tensorType);
            encodeBin(0, 12, tensorType);
        }

        encodeBin(signFlag, signCtx, tensorType);
    }

    encodeBin(sigFlag, sigCtx, tensorType);
}

void ANSEncoder::encodeAbsRem(const int32_t value, const uint16_t k) {
    uint8_t minusBits = 0;

    const uint32_t bitwidth = getBitwidthFromEnum(tensorBitwidth);
    if (bitwidth < 2) {
        // Hmmm, always false?
        encodeBinsEP(value, bitwidth);
        return;
    }
    uint32_t msb1 = (value >> (bitwidth - 1)) & 0x1;
    uint32_t msb2 = (value >> (bitwidth - 2)) & 0x1;

    minusBits += 2;

    uint32_t msb3 = 0, msb4 = 0, msb5 = 0, msb6 = 0;

    if (tensorBitwidth == TensorBitwidth::BW_12) {
        msb3 = (value >> (bitwidth - 3)) & 0x1;
        msb4 = (value >> (bitwidth - 4)) & 0x1;
        minusBits += 2;
    } else if (tensorBitwidth >= TensorBitwidth::BW_16) {
        msb3 = (value >> (bitwidth - 3)) & 0x1;
        msb4 = (value >> (bitwidth - 4)) & 0x1;
        msb5 = (value >> (bitwidth - 5)) & 0x1;
        msb6 = (value >> (bitwidth - 6)) & 0x1;
        minusBits += 4;
    }

    const uint32_t baseMask = (1 << (bitwidth - minusBits)) - 1;
    const uint32_t value_no_msb = value & baseMask;

    const uint8_t k_upd = k + 1;
    const uint32_t q = value_no_msb >> k_upd;
    const uint32_t r = value_no_msb & ((1 << k_upd) - 1);

    // suffix
    encodeBinsEP(r, k_upd);
    encodeBinEP(0);
    // unary prefix
    for (uint32_t i = 0; i < q; i++) {
        encodeBinEP(1);
    }


    if (tensorBitwidth == TensorBitwidth::BW_12) {
        encodeBin(msb4, 9, tensorType);
        encodeBin(msb3, 8, tensorType);
    } else if (tensorBitwidth >= TensorBitwidth::BW_16) {
        encodeBin(msb6, 11, tensorType);
        encodeBin(msb5, 10, tensorType);
        encodeBin(msb4, 9, tensorType);
        encodeBin(msb3, 8, tensorType);
    }

    encodeBin(msb2, 7, tensorType);
    encodeBin(msb1, 6, tensorType);
}

void ANSEncoder::encodeWeightsChunks(const int32_t *pWeights, const uint32_t numWeights) {
    const uint32_t width = getBitwidthFromEnum(tensorBitwidth);
    const uint32_t numChunks = (numWeights + chunkSize - 1) >> 11;
    std::vector<int32_t> scaledBuf(chunkSize);
    for (auto c = static_cast<int32_t>(numChunks - 1); c >= 0; c--) {
        contextModeler.resetNeighborCtx();

        const uint32_t start = static_cast<uint32_t>(c) * chunkSize;
        const uint32_t end = std::min(start + chunkSize, numWeights);
        const uint32_t len = end - start;

        int64_t sumRes = 0;
        int32_t residual = 0;

        // ---- pass 1:compute local mean ----
        int64_t sum = 0;
        for (uint32_t i = start; i < end; i++) {
            sum += pWeights[i];
        }

        const uint32_t shift = std::ceil(std::log2(len));
        int32_t localMean = sum >> shift;

        const bool useMean = (std::abs(localMean) > 4);
        // ---------- pass 2: residual + meanResidual -------------
        if (useMean) {
            for (uint32_t i = start; i < end; i++) {
                residual = pWeights[i] - localMean;
                sumRes += std::abs(residual);
            }
        } else {
            localMean = 0; // set local mean = 0
            for (uint32_t i = start; i < end; i++) {
                residual = pWeights[i];
                sumRes += std::abs(residual);
            }
        }

        // --------------- mean abs residual ---------------
        int32_t meanRes = sumRes / len;
        if (meanRes == 0) {
            meanRes = 1;
        }

        uint8_t k = 0;
        if (meanRes < 8) {
            k = 0;
        } else if (meanRes < 32) {
            k = 1;
        } else if (meanRes < 256) {
            k = 2;
        } else {
            k = 3;
        }

        // ------------ pass 3 - histogram on scaled residuals
        double estBits = 0;
        for (uint32_t i = start; i < end; i++) {
            const int32_t res = pWeights[i] - localMean;
            scaledBuf[i - start] = res; // store scaled residual for later encoding

            /// compute bins per element (rough bit estimation)
            const uint32_t absScaled = std::abs(res);
            if (absScaled == 0) {
                estBits += 1; // sig only (minimal)
            } else if (absScaled <= 5) {
                estBits += 1 + 1 + absScaled; // sig + sign + branch + grXFlags (branch included in absscaled)
            } else {
                // rough estimate: MSBs + 1 unary + k + suffix
                // here we can use xEncRemAbs logic without actual bin encoder calls
                uint32_t minusBits = 2; // first 2 MSBs
                if (width == 12) minusBits += 2;
                else if (width >= 16) minusBits += 4;
                const uint32_t remAbs = absScaled - 5;
                const uint32_t q = remAbs >> (k + 1);
                //uint32_t r = remAbs & ((1 << (k+1)) - 1);
                estBits += minusBits + 1 + q + 1 + (k + 1); // MSBs + branch + unary + 0 term + suffix
            }
        }
        estBits = round(estBits * 0.9);
        const double binsPerElement = estBits / len;
        const double normBPE = binsPerElement / width;
        const bool skipChunk = (normBPE > 0.98);

        if (skipChunk) {
            for (auto j = static_cast<int32_t>(end - 1); j >= static_cast<int32_t>(start); j--) {
                iae_v(width, pWeights[j]);
            }
            encodeBinEP(1); // skip chunk = true
            continue;
        }

        auto neighborIdx = static_cast<int32_t>((end - start - 1) - 1);
        for (auto i = static_cast<int32_t>(end - 1); i >= static_cast<int32_t>(start); i--) {
            if (neighborIdx < 0) {
                contextModeler.updateNeighborCtx(0);
            } else {
                contextModeler.updateNeighborCtx(scaledBuf[neighborIdx]);
            }
            neighborIdx--;
            const int32_t scaled = scaledBuf[static_cast<uint32_t>(i) - start];
            encodeWeightBAC(scaled, k);
        }

        uae_v(2, k);
        if (useMean) {
            iae_v(width, localMean);
        }
        encodeBinEP(useMean ? 1 : 0);
        encodeBinEP(0); // skip chunk = false
    }
}

void ANSEncoder::encodeTensorHeader(const uint32_t *shape,
                                    const uint32_t numDims, const uint16_t tensorId) {
    for (int32_t i = static_cast<int32_t>(numDims - 1); i >= 0; i--) {
        const uint32_t dimSize = shape[i];
        const int bitlen = (dimSize == 0) ? 1 : 32 - __builtin_clz(dimSize);
        encodeBinsEP(dimSize, bitlen);
        encodeBinsEP(bitlen - 1, 5);
    }
    encodeBinsEP(numDims, 3);
    encodeBinsEP(static_cast<uint32_t>(tensorBitwidth), 3);
    encodeBinEP(static_cast<uint8_t>(tensorType));
    encodeBinsEP(tensorId, MAX_TENSORS_BITS);
}

void ANSEncoder::encodeWeights(const int32_t *pWeights, const uint32_t numWeights) {
    encodeWeightsChunks(pWeights, numWeights);
}

std::vector<uint8_t> &ANSEncoder::finishEncoding() {
    const uint8_t offset = writer.flush();
    writer.bitstream.push_back(offset);
    for (auto &contextualizedState: contextualizedStates) {
        for (const uint8_t state : contextualizedState) {
            writer.bitstream.push_back(state);
        }
    }
    return writer.bitstream;
}
