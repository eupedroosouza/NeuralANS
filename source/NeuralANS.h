#pragma once
#include <cstdint>
#include <vector>

#include "Lib/CommonLib/Context.h"
#include "Lib/CommonLib/TypeDef.h"
#include "Lib/DecLib/ANSDecoder.h"
#include "Lib/EncLib/ANSEncoder.h"

class Encoder {
    ANSEncoder encoder = {};

public:
    Encoder() = default;

    /**
     * Create Encoder
     * @param context loaded context (based on tables)
     */
    explicit Encoder(const Context &context);

    ~Encoder() = default;

    /**
     * Encode a signed int until 32 bits (equiprobable)
     * @param bitwidth bitwidth of integer
     * @param value value to encode
     */
    void iae_v(uint8_t bitwidth, int32_t value);

    /**
     * Encode a unsigned int util 32 bits (equiprobable)
     * @param bitwidth bitwidth of integer
     * @param value value to encode
     */
    void uae_v(uint8_t bitwidth, uint32_t value);

    /**
     * Encode layer
     * @param tensor tensor to encode
     * @param tensorId id of tensor
     */
    void encodeLayer(const TensorMeta &tensor, uint16_t tensorId);

    /**
     * Encode model
     * @param modelTensors loaded tensors
     * @return encoded bytestream
     */
    std::vector<uint8_t> encodeModel(const std::vector<TensorMeta> &modelTensors);

    /**
     * Finish encoding (it's mandatory)
     * @return data ready to save
     */
    std::vector<uint8_t> &finishEncoding();
};

class Decoder {
    ANSDecoder decoder;

public:
    Decoder() = default;

    explicit Decoder(const Context &context, std::vector<uint8_t> data);

    ~Decoder() = default;

    int32_t iae_v(uint8_t bitwidth);

    uint32_t uae_v(uint8_t bitwidth);

    void decodeLayer(TensorMeta &tensor);

    void decodeModel(std::vector<TensorMeta> &modelTensors);

    uint32_t finishDecoding();
};
