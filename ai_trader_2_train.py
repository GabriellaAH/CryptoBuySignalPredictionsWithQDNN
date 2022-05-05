from agent import Agent
import numpy as np
import environ
from utils import plot_learning, plot_chart, save_to_csv
import tensorflow as tf
import pandas as pd
from finta import TA

def addextra(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = ["time", "open", "high", "low", "close", 'volume']
    df.index.names = ["date"]
    df = df.sort_index()
    df['ema100'] = TA.EMA(df, period=200)
    # df['ema100'] = TA.VWAP(df) 
    df['macd'] = TA.MACD(df)['MACD']
    df['signal'] = TA.MACD(df)['SIGNAL']
    df['indicator'] = df['macd'] - df['signal']
    df['mfm'] = ((df['close']-df['low'])-(df['high']-df['close']))/ np.round(np.where((df['high']-df['low']) > 0, df['high']/df['low'], 1), 1)  # (df['high']-df['low'])
    df['mfv'] = df['mfm'] * df['volume']
    df['cmf'] = df['mfm'].rolling(window=21).sum() / df['mfv'].rolling(window=21).sum()    
    df['atr'] = TA.ATR(df)
    df['rsi'] = TA.RSI(df)
    df['sma'] = TA.SMA(df, period=1440)
    df['adx'] = TA.ADX(df)
    df['adxl'] = TA.ADX(df, period=288)
    df['stochk'] = TA.STOCH(df)
    df['stochd'] = TA.STOCHD(df)
    df['bbup'] = TA.BBANDS(df)['BB_UPPER']
    df['bblow'] = TA.BBANDS(df)['BB_LOWER']
    df['pbbup'] = (df['bbup'] - df['close']) / df['close']
    df['pbblow'] = (df['bblow'] - df['close']) / df['close']    
    df['popen'] = df["open"].pct_change()
    df['phigh'] = df["high"] / df["close"]
    df['plow'] = df["low"] / df["close"]
    df['pclose'] = df["close"].pct_change()
    df['pvolume'] = df["volume"].pct_change()
    df['pmacd'] = df["macd"].pct_change()
    df['psignal'] = df["signal"].pct_change()
    df = df.drop(columns=['mfm', 'mfv'], axis= 1)
    df = df.iloc[300:, :]    
    return df

def validate(agent: Agent, env:environ.TradeEnv, label:str) -> float:
    print('Model validating (%s)...' % label)
    VALIDATE_NO = 100
    wins = 0
    losts = 0
    scores = 0
    rewards = 0 
    category = 'validate_' + label
    stpes_count = 0
    for i in range(VALIDATE_NO):
        done = False
        observation = env.reset()
        steps = 0
        while not done:
            action = agent.choose_action(observation, False)
            observation, reward, done, info = env.step(action)
            rewards += reward
            stpes_count += 1
            steps += 1
            if steps > 1000:
                done = True
                losts += 1
                return -1
        winlost = info['winlost']
        profit = info['profit']        
        if winlost == 1:
            wins += 1
        if winlost == -1:
            losts += 1
        scores += profit
    agent.log(
        [category+'/win', category+'/lost',category+'/ratio', category+'/profit', category+'/reward', category+'/avg_steps'],
        [wins, losts, wins/(wins+losts), scores/VALIDATE_NO, rewards/VALIDATE_NO, stpes_count/VALIDATE_NO]
    )
    wins=0
    losts=0
    print('Model validate profit: %.2f reward: %.2f' % (scores/VALIDATE_NO, rewards/VALIDATE_NO))
    return scores/VALIDATE_NO


if __name__ == '__main__':
    print(tf.test.gpu_device_name())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    tf.compat.v1.disable_eager_execution()
    
    # code snipet from: https://www.codegrepper.com/code-examples/python/frameworks/file-path-in-python/check+tensorflow+gpu
    print(tf.__version__)
    
    data = pd.read_csv('./data/BTC_19_22_5m.csv', index_col="startTime")
    data = addextra(data)
    print('train data loaded')
    data_test = pd.read_csv('./data/ETH_19_22_5m.csv', index_col="startTime")
    data_test = addextra(data_test)
    print('test data loaded')
    CANDLES = 72
    env = environ.TradeEnv(data=data, candles= CANDLES)
    env_test = environ.TradeEnv(data=data_test, candles= CANDLES)

    N_TRADE = 200000
    RENDER_MODE = 'computer'
    ACTIVATION = 'relu'
    GAMMA = 0.8
    LR = 0.1
    EPSILON_DEC = 1e-5
    EPSILON_START = 1
    EPSILON_END = 0.05
    
    agent = Agent(gamma=GAMMA, epsilon=EPSILON_START, lr=LR, 
                input_dims=env.observation_space.shape, epsilon_dec=EPSILON_DEC,
                n_actions=env.action_space.n, mem_size=1000000, batch_size=1,
                epsilon_end=EPSILON_END, fname='model1.tf_mod', activation=ACTIVATION)
    scores = []
    eps_history = []
    steps = []
    losses = []
    best_validated_profit = -1

    # agent.load_model()
    
    wins = 0
    losts = 0    
    for i in range(N_TRADE):
        done = False
        score = 0
        observation = env.reset()
        action = None
        observation_ = None
        stpes_count = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            stpes_count = info['steps']
            observation = observation_
            loss = agent.learn()
            if loss == None:
                agent.log(['learning/reward', 'learning/epsilon'],[reward, agent.epsilon])
            else:
                agent.log(['learning/loss', 'learning/reward', 'learning/epsilon'],[loss, reward, agent.epsilon])
            losses.append(loss)
            env.render(mode=RENDER_MODE)
        winlost = info['winlost']
        if winlost == 1:
            wins += 1
        if winlost == -1:
            losts += 1
        if wins+losts >= 100:
            agent.log(['performance/win', 'performance/lost','performance/ratio'],[wins, losts, wins/(wins+losts)])
            wins=0
            losts=0

        eps_history.append(agent.epsilon)
        scores.append(score)
        steps.append(stpes_count)
        profit = info['profit']
        agent.log(['performance/steps//trade', 'performance/profit'], [stpes_count, profit])
        if not i % 50:
            avg_score = np.mean(scores[-100:])
            print('Agent episode: ', i, 'score %.2f' % score,
                    'average_score %.2f' % avg_score,
                    'epsilon %.6f' % agent.epsilon,
                    'loss: %.6f' % loss)
        if not i % 1000 and i>20000:
            profit = validate(agent=agent, env=env_test, label='Test')
            if profit > best_validated_profit:
                best_validated_profit = profit
                agent.q_eval.save(('autosave/best_model_%.5f' % profit))
            profit = validate(agent=agent, env=env, label='Train')
                
    filename1 = 'agent.png'
    
    x = [i+1 for i in range(N_TRADE)]
    plot_learning(x, scores, eps_history, filename1)
    plot_chart(x,steps, 'Steps', 'steps.png')
    plot_chart(x,losses, 'Loss - Agent', 'loss.png')
    save_to_csv(scores, 'scores.csv')
    agent.save_model()
