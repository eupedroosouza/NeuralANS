#pragma once
#include <cstdint>

#include "../Utils/Bitstream.h"

class BinaryEquiprobableANS {
public:

    /**
    * The first state to that table (initialize you state variable with that value)
    * @return the first state
    */
    static  uint8_t getFirstState() {
        return 2;
    }

    /**
     * Encoded equiprobable symbol and write on bitstream
     * @param currentState the current state
     * @param symbol the symbol will be encoded
     * @param writer the bitstream writer
     * @return  new state (as unsigned 16 bits)
     */
    static void encode(uint8_t &currentState, const uint8_t symbol, BitstreamWriter &writer) {
        while (currentState >= 2) {
            const auto bit = static_cast<uint8_t>(currentState % 2);
            writer.write(1, bit);
            currentState /= 2;
        }

        currentState = (currentState * 2) + symbol;
    }

    /**
     * Decoded equiprobable symbol
     * @param currentState the current state
     * @param reader the bitstream reader
     * @return a pair with previous state and decoded symbol
     */
    static uint8_t decode(uint8_t &currentState, BitstreamReader &reader) {
        const auto symbol = static_cast<int8_t>(currentState % 2);
        currentState = static_cast<uint8_t>(currentState / 2);

        while (currentState < 2) {
            const uint8_t bit = reader.advance(1);
            currentState = static_cast<uint8_t>((currentState * 2) + bit);
        }

        return symbol;
    }
};
