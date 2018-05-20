import pandas as pd
import time
from poloniex import Poloniex

end = int(time.time())
start = end - 100000 * 1800
start, end

polo = Poloniex()
df = pd.DataFrame(polo.returnChartData("USDT_BTC", period=1800, start=start, end=end))
df['date'] = pd.to_datetime(df['date'], unit='s')


print (df)
