import matplotlib.pyplot as plt
from numpy import arange, vstack


def predict_10_years(y_train, y_test, predict, time_step):
    
    years = arange(1950+time_step, 2012)

    y1 = vstack((y_train, y_test))
    y2 = vstack((y_train, predict))
    
    plt.plot(years, y2, color='red', label='Forecast')
    plt.plot(years, y1, color='black', label='Real')
    plt.legend()
    plt.xlabel('years')
    plt.ylabel('TFP')
    plt.axvline(x=2001, linestyle='--')
    plt.show()