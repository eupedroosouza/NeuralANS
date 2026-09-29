#pragma once
#include <algorithm>
#include <cstdint>
#include <list>
#include <map>
#include <string>
#include <utility>

#include "TypeDef.h"
#include "../Utils/Bitstream.h"


class StateBitstream {
public:
    uint8_t size = 0;
    uint8_t bitstream = 0;

    StateBitstream() = default;

    explicit StateBitstream(const uint8_t size, const uint8_t bitstream) : size(size), bitstream(bitstream) {
    }

    ~StateBitstream() = default;
};

class State {
public:
    uint8_t state = 0;
    std::vector<uint8_t> nextStates = {};
    std::vector<StateBitstream> bitstreams = {};

    State() = default;

    explicit State(const uint8_t state, const std::vector<uint8_t> &nextStates,
                   const std::vector<StateBitstream> &bitstreams) : state(state), nextStates(nextStates),
                                                                    bitstreams(bitstreams) {
    }

    virtual ~State() = default;
};

class DecodeState {
public:
    std::int8_t symbol = -1;
    std::uint8_t N = 0;
    std::uint8_t base = 0;

    DecodeState() = default;

    explicit DecodeState(const std::int8_t symbol, const std::uint8_t N, const std::uint8_t base) : symbol(symbol),
        N(N),
        base(base) {
    }

    ~DecodeState() = default;
};


class Table {
    uint16_t L = 0;

    std::vector<State> states = {};
    std::vector<DecodeState> decodeStates = {};

public:
    Table() = default;

    explicit Table(const uint16_t L, const std::list<State> &states) {
        this->L = L;

        this->states.resize(L);
        for (const State &state: states) {
            this->states[state.state] = state;
        }

        this->decodeStates.resize(L);
        for (const auto &state: this->states) {
            for (int8_t symbol = 0; symbol < static_cast<int8_t>(state.nextStates.size()); ++symbol) {
                const uint8_t &nextState = state.nextStates[symbol];
                if (state.bitstreams.empty() && this->decodeStates[nextState].symbol == -1) {
                    this->decodeStates[nextState] = DecodeState(symbol, 0, state.state);
                    continue;
                }
                const StateBitstream &stateBitstream = state.bitstreams.at(symbol);
                if (this->decodeStates[nextState].symbol == -1) {
                    this->decodeStates[nextState] = DecodeState(symbol, stateBitstream.size, state.state);
                } else {
                    DecodeState &decodeState = this->decodeStates[nextState];
                    decodeState.base = std::min(decodeState.base, state.state);
                }
            }
        }
    }

    virtual ~Table() = default;

    /**
     * The first state to that table (initialize you state variable with that value)
     * @return the first state
     */
    [[nodiscard]] uint8_t getFirstState() const {
        return 0;
    }

    /**
     * Encoded symbol and write on bitstream
     * @param currentState the current state
     * @param symbol the symbol will be encoded
     * @param writer the bitstream writer
     * @return new state (as unsigned 16 bits)
     */
    void encode(uint8_t &currentState, const int8_t symbol, BitstreamWriter &writer) const {
        const State &state = states[currentState];
        const uint8_t &nextState = state.nextStates.at(symbol);
        const StateBitstream &bitstream = state.bitstreams.at(symbol);
        writer.write(bitstream.size, bitstream.bitstream);
        currentState = nextState;
    }

    /**
     * Decode the simbol and write on bitstream
     * @param currentState the current state
     * @param reader the bitstream reader
     * @return a pair of previous state and symbol value
     */
    uint8_t decode(uint8_t &currentState, BitstreamReader &reader) const {
        const DecodeState &decodeState = this->decodeStates[currentState];

        // A state without a bitstream
        if (decodeState.N == 0) {
            currentState = decodeState.base;
            return decodeState.symbol;
        }

        const auto readBitstreams = reader.advance(decodeState.N);
        currentState = decodeState.base + readBitstreams;
        return decodeState.symbol;
    }
};
