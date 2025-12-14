import random

class StrategyBase:
    def __init__(self, t_current, p_current, position, state):
        self.recommendation = 0
        flag = self._condition(t_current, p_current, state)
        if flag:
            trend = state.trend
            self._update(position, trend)

    def _condition(self, t_current, p_current, state):
        pass

    def _update(self, position, trend):
        if trend == 'downtrend' and not position:
            self.recommendation = 1  # Buy
        elif trend == 'uptrend' and position:
            self.recommendation = 2  # Sell

# St1
class St1(StrategyBase):
    def _condition(self, t_current, p_current, state):
        theta = state.theta
        p_ext = state.p_ext
        delta_p_relative = abs(p_current - p_ext) / p_ext
        return delta_p_relative >= 2 * theta

# St2
class St2(StrategyBase):
    def _condition(self, t_current, p_current, state):
        dc_duration = state.dc_duration
        t_dcc = state.t_dcc
        t_os = t_current - t_dcc
        return t_os >= 2 * dc_duration

# St3
class St3(StrategyBase):
    def _condition(self, t_current, p_current, state):
        p_dcc_star = state.p_dcc_star
        theta = state.theta
        osv_cur = abs((p_current - p_dcc_star) / (theta * p_dcc_star))
        osv_best = state.osv_best
        return osv_cur >= osv_best

# St4
class St4(StrategyBase):
    def _condition(self, t_current, p_current, state):
        theta = state.theta
        p_ext_initial = state.p_ext_initial
        tmv_cur = abs((p_current - p_ext_initial) / (theta * p_ext_initial))
        tmv_best = state.tmv_best
        return tmv_cur >= tmv_best

# St5
class St5(StrategyBase):
    def _condition(self, t_current, p_current, state):
        dc_duration = state.dc_duration
        t_dcc = state.t_dcc
        t_os = t_current - t_dcc
        ratio_os_dc = t_os / dc_duration if dc_duration > 0 else 0
        return ratio_os_dc >= state.rd

# St6
class St6(StrategyBase):
    def _condition(self, t_current, p_current, state):
        if state.is_dcc:
            if state.trend == 'downtrend':
                r = random.uniform(0, 1)
                return r >= state.rn
            elif state.trend == 'uptrend':
                return True
        return False

# St7
class St7(StrategyBase):
    def _condition(self, t_current, p_current, state):
        if len(state.trend_history) >= 5:
            seq = state.trend_history[-5:]
            os_seq = state.if_os[-5:]
            if seq == ['uptrend', 'downtrend', 'uptrend', 'downtrend', 'uptrend'] and all(not os for os in os_seq[1::2]) and all(os for os in os_seq[0::2]):
                if state.trend == 'uptrend':
                    return True
                elif state.trend == 'downtrend' and state.is_dcc:
                    return True
        return False

    def _update(self, position, trend):
        if trend == 'uptrend' and not position:
            self.recommendation = 1
        elif trend == 'downtrend' and position:
            self.recommendation = 2  # Sell in DT

# St8
class St8(StrategyBase):
    def _condition(self, t_current, p_current, state):
        if len(state.trend_history) >= 5:
            seq = state.trend_history[-5:]
            os_seq = state.if_os[-5:]
            if seq == ['downtrend', 'uptrend', 'downtrend', 'uptrend', 'downtrend'] and all(not os for os in os_seq[1::2]) and all(os for os in os_seq[0::2]):
                if state.trend == 'downtrend':
                    return True
                elif state.trend == 'uptrend' and state.is_dcc:
                    return True
        return False