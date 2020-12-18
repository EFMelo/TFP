from pandas import read_csv
from numpy import array
from sklearn.preprocessing import StandardScaler

class TFP:
    
    @classmethod
    def load_data(cls, path, time_step, country):
        
        norm = StandardScaler()
        
        # Loading data
        data = read_csv(path)

        if country == 'USA':
            data2 = data[data['isocode'] == 'USA'].rtfpna.values  # USA data
            data2 = norm.fit_transform(data2.reshape(-1, 1))
        elif country == 'CAN':
            data2 = data[data['isocode'] == 'CAN'].rtfpna.values  # CAN data
        elif country == 'MEX':
            data2 = data[data['isocode'] == 'MEX'].rtfpna.values  # MEX data
        
        x, y = cls.__input_output(data2, time_step)  # input and output
        
        # Training and Testing data
        x_train, x_test = x[0:data2.shape[0]-time_step-10], x[data2.shape[0]-time_step-10:data2.shape[0]-time_step]
        y_train, y_test = y[0:data2.shape[0]-time_step-10], y[data2.shape[0]-time_step-10:data2.shape[0]-time_step]
        
        return x_train, y_train, x_test, y_test, norm
    
    
    @classmethod
    def __input_output(cls, data, time_step):
        
        """
        Creates the input and output of each series
        """
        
        x = []
        y = []
        
        for i in range(time_step, data.shape[0]):
            x.append(data[i-time_step:i])
            y.append(data[i])
        
        x, y = array(x), array(y)
        
        return x.reshape(x.shape[0], x.shape[1], 1), y.reshape(-1, 1)