from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import RMSprop
 
class Models:

    @classmethod
    def model_usa(cls, x):
        
        model = Sequential()
        
        # LSTM layers
        model.add(LSTM(units=18, return_sequences=True, input_shape=(x.shape[1], 1)))
        model.add(Dropout(0.12059804609229516))
        
        model.add(LSTM(units=18, return_sequences=True))
        model.add(Dropout(0.12059804609229516))

        model.add(LSTM(units=18))
        model.add(Dropout(0.12059804609229516))
        
        # Output
        model.add(Dense(units=1, activation='linear'))
        
        # Compile
        model.compile(optimizer=RMSprop(lr=1e-4), loss='mean_squared_error')
        
        return model


    @classmethod
    def model_can(cls, x):
        
        model = Sequential()
        
        # LSTM layers
        model.add(LSTM(units=22, input_shape=(x.shape[1], 1)))
        model.add(Dropout(0.3272439609130082))
        
        # Output
        model.add(Dense(units=1, activation='linear'))
        
        # Compile
        model.compile(optimizer=RMSprop(lr=8.008399440689927e-05), loss='mean_squared_error')
        
        return model


    @classmethod
    def model_mex(cls, x):

        model = Sequential()
        
        # LSTM layers
        model.add(LSTM(units=48, input_shape=(x.shape[1], 1)))
        model.add(Dropout(0.3422611380284145))
        
        # Output
        model.add(Dense(units=1, activation='linear'))
        
        # Compile
        model.compile(optimizer=RMSprop(lr=4.0261752342267e-05), loss='mean_squared_error')
        
        return model