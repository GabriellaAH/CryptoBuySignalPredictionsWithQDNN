# from turtle import color
import gym
import gym.spaces
import numpy
from gym.utils import seeding
from gym.envs.registration import EnvSpec
import enum
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
import pandas as pd
import random

class Actions(enum.Enum):    
    Hold = 0
    Buy1 = 1 # sl: atr2 tp: atr3 RiskReward: 1:1.5
    # Buy2 = 2 # sl: atr2 tp: atr4 RR: 1:2
    # Buy3 = 3 # sl: atr2 tp: atr6 RR: 1:3

class State:
    def __init__(self, data:pd.DataFrame, candles:int):
        assert isinstance(data, pd.DataFrame)
        assert isinstance(candles, int)
        self.data = data
        self.candles = candles
        self.step_count = 0
        self.offset = 0
        self.haveposition = False
        self.open_price = 0
        self.sl = 0
        self.tp = 0
        self.index_labels = self.data.index.tolist()
        self.buy_type = 0
        self.start_offset = 0
        self.win_lost = 0
        self.profit = 0
        self.reset()
        
        
        
    def reset(self):
        self.step_count = 0
        self.offset = random.randint(self.candles+1500, len(self.data) - 1000)
        self.start_offset = self.offset
        self.haveposition = False
        self.open_price = 0
        self.sl = 0
        self.tp = 0
        self.buy_type = 0
        self.win_lost = 0
        self.profit = 0

    @property
    def shape(self):
        # close, volume
        return self.candles * 14 + 11,

    def encode(self):
        """
        2D -> 1D conversion 
        Convert current state into numpy array.  
        """
        src = self.data
        index_labels = self.index_labels    
        datas = []
        # if self.haveposition:
        #     # datas.append(numpy.float32(1))
        #     datas.append(numpy.float32(-0.1))
        # else:
        #     # datas.append(numpy.float32(0))
        #     datas.append(numpy.float32((self.offset-self.start_offset)/1000))
        cmf = src.loc[index_labels[self.offset], 'cmf'] 
        if cmf > 0:
            datas.append(numpy.float32(1))
        else:
            datas.append(numpy.float32(0))
            
        indicator = src.loc[index_labels[self.offset], 'indicator'] 
        prev_indicator = src.loc[index_labels[self.offset-1], 'indicator'] 
        if indicator > 0 and prev_indicator < 0:
            datas.append(numpy.float32(1))
        else:
            datas.append(numpy.float32(0))
            
        rsi = src.loc[index_labels[self.offset], 'rsi']
        prev_rsi = src.loc[index_labels[self.offset-1], 'rsi']
        if rsi < 30:
            datas.append(numpy.float32(1))
        else:
            datas.append(numpy.float32(0))
        if prev_rsi < 30 and rsi > 30:
            datas.append(numpy.float32(1))
        else:
            datas.append(numpy.float32(0))

        closep = src.loc[index_labels[self.offset], 'close']
        ema = src.loc[index_labels[self.offset], 'ema100']
        sma = src.loc[index_labels[self.offset], 'sma']
        if closep > ema:
            datas.append(numpy.float32(1))
        else:
            datas.append(numpy.float32(0))
        if closep > sma:
            datas.append(numpy.float32(1))
        else:
            datas.append(numpy.float32(0))
        if ema > sma:
            datas.append(numpy.float32(1))
        else:
            datas.append(numpy.float32(0))            
        adx = src.loc[index_labels[self.offset], 'adx']
        adxl = src.loc[index_labels[self.offset], 'adxl']
        if adx > 50:
            datas.append(numpy.float32(1))
        else:
            datas.append(numpy.float32(0))
        if adxl > 50:
            datas.append(numpy.float32(1))
        else:
            datas.append(numpy.float32(0))
        stochk = src.loc[index_labels[self.offset], 'stochk']
        stochd = src.loc[index_labels[self.offset], 'stochd']
        prev_stochk = src.loc[index_labels[self.offset-1], 'stochk']
        prev_stochd = src.loc[index_labels[self.offset-1], 'stochd']
        if stochk < 20 and stochd < 20:
            datas.append(numpy.float32(1))
            if prev_stochd > prev_stochk and stochd < stochk:
                datas.append(numpy.float32(1))
            else:
                datas.append(numpy.float32(0))
        else:
            datas.append(numpy.float32(0))
            datas.append(numpy.float32(0))
                    
        
        for x in range(self.offset-self.candles+1, self.offset+1):
            closep = src.loc[index_labels[x], 'pclose'] /10
            vol = src.loc[index_labels[x], 'pvolume'] / 1000000
            ema = src.loc[index_labels[x], 'close'] / src.loc[index_labels[x], 'ema100'] /10
            sma = src.loc[index_labels[x], 'close'] / src.loc[index_labels[x], 'sma'] /10
            macd = src.loc[index_labels[x], 'pmacd'] /10000
            signal = src.loc[index_labels[x], 'psignal'] /10000
            rsi = src.loc[index_labels[x], 'rsi']  / 100
            highp = src.loc[index_labels[x], 'phigh'] /10
            lowp = src.loc[index_labels[x], 'plow'] /10
            cmf = src.loc[index_labels[x], 'cmf'] / 100
            if cmf > 1:
                cmf = 1
            if cmf < -1:
                cmf = -1
            adx = src.loc[index_labels[x], 'adx'] / 100
            adxl = src.loc[index_labels[x], 'adxl'] / 100
            stochk = src.loc[index_labels[x], 'stochk'] / 100
            stochd = src.loc[index_labels[x], 'stochd'] / 100
            
            datas.append(numpy.float32(ema))
            datas.append(numpy.float32(sma))
            datas.append(numpy.float32(highp))
            datas.append(numpy.float32(lowp))
            datas.append(numpy.float32(closep))
            datas.append(numpy.float32(vol))
            datas.append(numpy.float32(macd))
            datas.append(numpy.float32(signal))
            datas.append(numpy.float32(rsi))
            datas.append(numpy.float32(cmf))
            datas.append(numpy.float32(adx))
            datas.append(numpy.float32(adxl))
            datas.append(numpy.float32(stochk))
            datas.append(numpy.float32(stochd))
            
        res = numpy.array(datas)   
        res[res == inf] = 0
        res[res == -inf] = 0
        res = numpy.nan_to_num(res, copy=True) 

        return res


    def get_action(self):
        return -1

    def is_winner(self, tp=None, sl=None):
        for i in range(self.offset+1, self.offset+50):
            minp = self.data.loc[self.index_labels[i], 'low'] 
            maxp = self.data.loc[self.index_labels[i], 'high'] 
            if tp != None and sl != None:
                if minp <= sl:
                    return False
                if maxp >= tp:
                    return True
            else:    
                if minp <= self.sl:
                    self.win_lost = -1
                    return False
                if maxp >= self.tp:
                    self.win_lost = 1
                    return True
        self.win_lost = 0
        return False
                
    def step(self, action):
        """
        Process the next step and calculate rewards
        :param action:
        :return: reward, done
        """
        assert isinstance(action, Actions)
        
        self.step_count += 1
        
        reward = 0
        done = False
        closep =  self.data.loc[self.index_labels[self.offset], 'close']
        # minp = self.data.loc[self.index_labels[self.offset], 'low'] 
        # maxp = self.data.loc[self.index_labels[self.offset], 'high'] 
        atr =  self.data.loc[self.index_labels[self.offset], 'atr']
        # ema = self.data.loc[self.index_labels[self.offset], 'ema100']         
        # sma = self.data.loc[self.index_labels[self.offset], 'sma']
        # rsi = self.data.loc[self.index_labels[self.offset], 'rsi'] 
        # indicator = self.data.loc[self.index_labels[self.offset], 'indicator'] 
        # indicator_prew = self.data.loc[self.index_labels[self.offset-1], 'indicator'] 
        
        # if self.haveposition:  
            # reward += -0.1
            # if minp <= self.sl:
            #     done |= True
            #     reward += -100                 
            # elif maxp >= self.tp:
            #     done |= True
            #     reward += 100 * self.buy_type
            # if action == Actions.Hold:
            #     reward += 1.1
        # else:
            # reward += -0.1
            # if self.offset - self.start_offset > 2000:
            #     reward += -50000
            #     done |= True
            # if self.offset - self.start_offset > 1000:
            #     reward += -50
            # if self.offset - self.start_offset > 500:
            #     reward += -20
            # if self.offset - self.start_offset > 250:
            #     reward += -10
            
        if not done and not self.haveposition:
            if action == Actions.Buy1:
                self.open_price = closep
                self.tp = closep + atr * 4
                self.sl = closep - atr * 2
                self.buy_type = 1
                self.haveposition = True
                if self.is_winner():
                    reward += 100 * 2
                    self.profit = 2
                else:
                    reward += -100
                    self.profit = -1
                done |= True
            if action == Actions.Hold:
                tp = closep + atr * 4
                sl = closep - atr * 2
                if self.is_winner(tp=tp, sl=sl):
                    reward += -1
                else:
                    reward += 1
                
            # if action == Actions.Buy2:
            #     self.open_price = closep
            #     self.tp = closep + atr * 4
            #     self.sl = closep - atr * 2
            #     self.buy_type = 2
            #     self.haveposition = True
            #     if self.is_winner():
            #         reward += 500 * 2
            #         self.profit = 2
            #     else:
            #         reward += -500
            #         self.profit = -1
            #     done |= True
            # if action == Actions.Buy3:
            #     self.open_price = closep
            #     self.tp = closep + atr * 6
            #     self.sl = closep - atr * 2
            #     self.buy_type = 3
            #     self.haveposition = True
            #     if self.is_winner():
            #         reward += 500 * 3
            #         self.profit = 3
            #     else:
            #         reward += -500
            #         self.profit = -1
            #     done |= True
            # if action == Actions.Buy1 or action == Actions.Buy2 or action == Actions.Buy3:
            #     if ema < closep:
            #         reward += 5
            #     if ema > sma:
            #         reward += 1
            #     if sma < ema < closep:
            #         reward += 5
            #     if rsi < 30:
            #         reward += 10
            #     if indicator > 0 and indicator_prew < 0:
            #         reward += 10
            #     if ema > closep:
            #         reward -= 10
            #     if rsi > 80:
            #         reward -= 10
            # if action == Actions.Hold:
            #     if closep < sma:
            #         reward += 0.05
            #     if closep < ema:
            #         reward += 0.05
            #     if sma < ema:
            #         reward += 0.01
            #     if rsi > 90:
            #         reward += 0.01

        done |= self.offset > len(self.data) -2
        self.offset += 1

        return reward, done

class TradeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    spec = EnvSpec("TraderEnv-v0")

    def __init__(self, data:pd.DataFrame, candles:int):
        self._state = State(data=data, candles=candles)
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=self._state.shape, dtype=np.float32)
        self.seed()  
        self._first_rendering = True      

    def reset(self):
        self._state.reset()
        return self._state.encode()

    def step(self, action_idx):
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {
            "steps": self._state.step_count,
            "winlost" : self._state.win_lost,
            "profit" : self._state.profit
        }
        return obs, reward, done, info

    def get_action(self):
        action = int(self._state.get_action())
        return action

    def render(self, mode='human', pause=False):
        pass        
        assert mode in ['human', 'computer']

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

