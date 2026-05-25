#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Daydream Handler – Risk Manager
// Position sizing, SL/TP placement (polygonal structure + pip constraints),
// drawdown limits, spread-aware execution.
// ─────────────────────────────────────────────────────────────────────────────
#include "../../data/market_data.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dh::risk {

struct RiskConfig {
    double account_equity   = 10000.0;  // account balance
    double risk_pct         = 0.01;     // 1% risk per trade
    double max_dd_pct       = 0.10;     // max 10% drawdown
    double min_tp_pips      = 1.0;      // minimum TP distance
    double max_tp_pips      = 80.0;     // maximum TP distance
    double max_spread_pips  = 3.0;      // reject if spread > this
    double max_positions    = 5.0;      // concurrent open trades
};

struct PositionSpec {
    double  entry_price;
    double  stop_loss;
    double  take_profit;
    double  lot_size;
    bool    is_long;
    double  risk_amount;
    double  reward_amount;
    double  rr_ratio;
};

class RiskManager {
public:
    explicit RiskManager(RiskConfig cfg = {}) : m_cfg(cfg) {}

    /// Compute a full position specification from entry + direction + range.
    /// sl_ref  = last swing high (short) or swing low (long)
    /// tp_ref  = Fibonacci / structural target price
    PositionSpec size_position(const data::Instrument& instr,
                                double entry_price,
                                double sl_ref,
                                double tp_ref,
                                bool   is_long) const
    {
        PositionSpec spec{};
        spec.entry_price = entry_price;
        spec.is_long     = is_long;

        // Place SL just behind last peak/deep with one pip buffer
        double pip   = instr.pip_size;
        double buf   = pip;  // 1 pip behind swing
        spec.stop_loss   = is_long  ? sl_ref - buf : sl_ref + buf;
        spec.take_profit = tp_ref;

        // Clamp TP to institutional range [min_tp, max_tp] pips
        double tp_pips = std::abs(spec.take_profit - entry_price) / pip;
        tp_pips = std::clamp(tp_pips, m_cfg.min_tp_pips, m_cfg.max_tp_pips);
        spec.take_profit = is_long
                           ? entry_price + tp_pips * pip
                           : entry_price - tp_pips * pip;

        // Risk in price terms
        double sl_dist   = std::abs(entry_price - spec.stop_loss);
        double tp_dist   = std::abs(spec.take_profit - entry_price);

        spec.rr_ratio     = sl_dist > 0 ? tp_dist / sl_dist : 0.0;
        spec.risk_amount  = m_cfg.account_equity * m_cfg.risk_pct;
        spec.reward_amount = spec.risk_amount * spec.rr_ratio;

        // Lot size = risk_amount / (sl_distance / pip * pip_value)
        double pip_value = instr.pip_size * instr.lot_size;
        double sl_pips   = sl_dist / pip;
        spec.lot_size    = sl_pips > 0
                           ? spec.risk_amount / (sl_pips * pip_value)
                           : 0.0;
        spec.lot_size = std::max(0.0, spec.lot_size);
        return spec;
    }

    /// Validate spread against max allowed
    bool spread_ok(const data::Instrument& instr, double live_spread_pips) const {
        return live_spread_pips <= m_cfg.max_spread_pips
               && live_spread_pips <= instr.spread * 3.0;
    }

    /// Check if adding a new trade would breach drawdown limits
    bool drawdown_ok(double current_equity) const {
        double dd = (m_cfg.account_equity - current_equity)
                    / m_cfg.account_equity;
        return dd < m_cfg.max_dd_pct;
    }

    const RiskConfig& config() const { return m_cfg; }
    void set_equity(double eq) { m_cfg.account_equity = eq; }

private:
    RiskConfig m_cfg;
};

} // namespace dh::risk
