#include "NeuralANS.h"


static constexpr uint32_t MAX_TENSOR_DIMS = 8; // max tensor rank supported


Encoder::Encoder(const Context &context) {
    this->encoder = ANSEncoder(context);
}

void Encoder::iae_v(const uint8_t bitwidth, const int32_t value) {
    encoder.iae_v(bitwidth, value);
}

void Encoder::uae_v(const uint8_t bitwidth, const uint32_t value) {
    encoder.uae_v(bitwidth, value);
}

void Encoder::encodeLayer(const TensorMeta &tensor, const uint16_t tensorId) {
    const uint32_t numWeights = tensor.data.size();

    encoder.setBitwidthAndType(tensor.tensorBitwidth, tensor.tensorType);

    encoder.encodeWeights(tensor.data.data(), numWeights);
    encoder.encodeTensorHeader(tensor.shape.data(), tensor.numDims, tensorId);
}

std::vector<uint8_t> Encoder::encodeModel(const std::vector<TensorMeta> &modelTensors) {
    const uint32_t numTensors = modelTensors.size();
    // Inversed to decode on order
    for (auto tensorId = static_cast<int16_t>(numTensors - 1); tensorId >= 0; tensorId--) {
        const auto normalizedTensorId = static_cast<uint16_t>(tensorId);
        encodeLayer(modelTensors[normalizedTensorId], normalizedTensorId);
    }
    uae_v(ANSEncoder::MAX_TENSORS_BITS, numTensors);
    return this->finishEncoding();
}

std::vector<uint8_t> &Encoder::finishEncoding() {
    return encoder.finishEncoding();
}

Decoder::Decoder(const Context &context, std::vector<uint8_t> data) {
    this->decoder = ANSDecoder(context, std::move(data));
}

int32_t Decoder::iae_v(const uint8_t bitwidth) {
    return this->decoder.iae_v(bitwidth);
}

uint32_t Decoder::uae_v(const uint8_t bitwidth) {
    return this->decoder.uae_v(bitwidth);
}

void Decoder::decodeLayer(TensorMeta &tensor) {
    uint32_t shape[MAX_TENSOR_DIMS] = {0};
    uint32_t numDims = 0;

    // Decode header
    decoder.decodeTensorHeader(shape, numDims, tensor);
    // Copy shape array into vector
    tensor.shape.assign(shape, shape + numDims);

    // Compute number of weights
    uint32_t numWeights = 1;
    for (uint32_t i = 0; i < numDims; i++)
        numWeights *= shape[i];

    // Resize tensor data to hold decoded weights
    tensor.data.resize(numWeights);


    // Decode weights
    decoder.decodeWeights(tensor.data.data(), numWeights);
}

void Decoder::decodeModel(std::vector<TensorMeta> &modelTensors) {
    const uint32_t numTensors = decoder.uae_v(ANSEncoder::MAX_TENSORS_BITS);
    modelTensors.resize(numTensors);
    for (uint32_t i = 0; i < numTensors; i++) {
        decodeLayer(modelTensors[i]);
    }
}
