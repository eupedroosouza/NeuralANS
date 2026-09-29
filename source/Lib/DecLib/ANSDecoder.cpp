#include "ANSDecoder.h"

#include "Lib/EncLib/ANSEncoder.h"

ANSDecoder::ANSDecoder(const Context &context, std::vector<uint8_t> bytestream) {
    this->context = context;
    // Recovery last state
    for (int i = 1; i >= 0; i--) {
        for (int j = 12; j >= 0; j--) {
            const uint8_t state = bytestream.back();
            bytestream.pop_back();
            contextualizedStates[i][j] = state;
        }
    }
    const uint8_t offset = bytestream.back();
    bytestream.pop_back();
    this->reader = BitstreamReader(std::move(bytestream), offset);
}

uint32_t ANSDecoder::decodeBin(const uint8_t ctxId, const TensorType paramType) {
    const Table *table = context.getContext(ctxId, paramType);
    const auto pType = static_cast<uint8_t>(paramType);
    return table->decode(this->contextualizedStates[pType][ctxId], this->reader);
}


uint32_t ANSDecoder::decodeBinEP() {
    return reader.advance(1);
}


uint32_t ANSDecoder::decodeBinsEP(const uint32_t numBins) {
    return reader.advance(numBins);
}

int32_t ANSDecoder::iae_v(const uint8_t bitwidth) {
    const uint32_t pattern = decodeBinsEP(bitwidth);
    return static_cast<int32_t>(pattern << (32 - bitwidth)) >> (32 - bitwidth);
}

uint32_t ANSDecoder::uae_v(const uint8_t bitwidth) {
    return decodeBinsEP(bitwidth);
}

void ANSDecoder::decodeTensorHeader(uint32_t *shape, uint32_t &numDims, TensorMeta &tensor) {
    tensor.tensorId = decodeBinsEP(ANSEncoder::MAX_TENSORS_BITS);
    tensorType = static_cast<TensorType>(decodeBinEP());
    tensor.tensorType = tensorType;

    tensorBitwidth = static_cast<TensorBitwidth>(decodeBinsEP(3));
    tensor.tensorBitwidth = tensorBitwidth;

    numDims = decodeBinsEP(3);
    tensor.numDims = numDims;

    uint32_t bitlenMinus1 = 0, bitlen = 0;
    for (uint32_t i = 0; i < numDims; i++) {
        bitlenMinus1 = decodeBinsEP(5);
        bitlen = bitlenMinus1 + 1;
        shape[i] = decodeBinsEP(bitlen);
    }
}

void ANSDecoder::decodeWeightsChunks(int32_t *pWeights, const uint32_t numWeights) {
    const uint32_t width = getBitwidthFromEnum(tensorBitwidth);

    constexpr uint32_t chunkSize = ANSEncoder::chunkSize;
    const uint32_t numChunks = (numWeights + chunkSize - 1) >> 11;

    for (uint32_t c = 0; c < numChunks; c++) {
        contextModeler.resetNeighborCtx();

        const uint32_t start = c * chunkSize;
        const uint32_t end = std::min(start + chunkSize, numWeights);


        const bool skipChunk = decodeBinEP();
        if (skipChunk) {
            /// raw ep bins decoding
            for (uint32_t i = start; i < end; i++) {
                pWeights[i] = iae_v(width);
            }
            continue;
        }

        const bool useMean = decodeBinEP();

        int32_t localMean = 0;
        if (useMean) {
            localMean = iae_v(width);
        }

        const uint8_t k = uae_v(2);

        for (uint32_t i = start; i < end; i++) {
            int32_t decodedVal = 0;
            decodeWeightVal(decodedVal, k);
            const int32_t residual = decodedVal;
            pWeights[i] = residual + localMean;
            contextModeler.updateNeighborCtx(decodedVal);
        }
    }
}

void ANSDecoder::decodeWeights(int32_t *pWeights, uint32_t numWeights) {
    return decodeWeightsChunks(pWeights, numWeights);
}

void ANSDecoder::decodeWeightVal(int32_t &decodedIntVal, const uint8_t k) {

    const uint32_t sigFlag = decodeBin(contextModeler.getSigCtxId(), tensorType); // sig (significant flag)

    decodedIntVal = 0;

    if (!sigFlag) {
        return;
    }

    //sign
    const int32_t signCtx = contextModeler.getSignFlagCtxId();
    const uint32_t signFlag = decodeBin(signCtx, tensorType);

    const uint32_t branchFlag = decodeBin(12, tensorType); // assuming context 8 is for branch flag
    if (branchFlag) {
        uint32_t remAbsLevel = 0;
        decodeAbsRem(remAbsLevel, k);
        decodedIntVal = signFlag ? -static_cast<int32_t>(remAbsLevel + 6) : static_cast<int32_t>(remAbsLevel + 6);
        return;
    }

    uint32_t remAbsLevel = 0;
    uint32_t grXFlag = 0;
    uint8_t numGreaterFlagsDecoded = 0;

    do {
        const uint32_t ctxIdx = contextModeler.getGtxCtxId(signFlag);
        grXFlag = decodeBin(ctxIdx, tensorType);
        if (grXFlag) {
            remAbsLevel++;
        }
        numGreaterFlagsDecoded++;
    } while (grXFlag && numGreaterFlagsDecoded < 4); // 4 basedo on SBAC

    decodedIntVal = static_cast<int32_t>(remAbsLevel) + 1; // add 1 to get the original abs value
    decodedIntVal = signFlag ? -decodedIntVal : decodedIntVal;
}

void ANSDecoder::decodeAbsRem(uint32_t &remainder, const uint32_t k) {
    const uint32_t bitwidth = getBitwidthFromEnum(tensorBitwidth);
    uint8_t plusBits = 0;

    if (bitwidth < 2) {
        remainder = decodeBinsEP(bitwidth);
        return;
    }

    // ---- 1. Decode MSBs (context-coded) ----
    const uint32_t msb1 = decodeBin(6, tensorType);
    const uint32_t msb2 = decodeBin(7, tensorType);
    plusBits += 2;

    uint32_t msb3 = 0, msb4 = 0, msb5 = 0, msb6 = 0;
    if (tensorBitwidth == TensorBitwidth::BW_12) {
        msb3 = decodeBin(8, tensorType);
        msb4 = decodeBin(9, tensorType);
        plusBits += 2;
    } else if (tensorBitwidth >= TensorBitwidth::BW_16) {
        msb3 = decodeBin(8, tensorType);
        msb4 = decodeBin(9, tensorType);
        msb5 = decodeBin(10, tensorType);
        msb6 = decodeBin(11, tensorType);
        plusBits += 4;
    }


    // ---- 2. Decode unary prefix ----
    uint32_t q = 0;
    const uint8_t k_upd = k + 1;


    while (true) {
        const uint32_t bin = decodeBinEP();
        if (bin == 0)
            break;
        q++;
    }
    // printf("(d) q: %d\n", q);

    // ---- 3. Decode suffix ----
    const uint32_t r = decodeBinsEP(k_upd);

    // ---- 4. Reconstruct value from MSBs, unary prefix, and suffix ----
    const uint32_t lowerMask = (1u << (bitwidth - plusBits)) - 1;
    uint32_t lower = (q << k_upd) | r;
    lower &= lowerMask;

    const uint32_t value = (msb1 << (bitwidth - 1)) |
                           (msb2 << (bitwidth - 2)) |
                           (msb3 << (bitwidth - 3)) |
                           (msb4 << (bitwidth - 4)) |
                           (msb5 << (bitwidth - 5)) |
                           (msb6 << (bitwidth - 6)) |
                           lower;

    remainder = value;
}
