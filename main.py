from dataset import TFP
from models import Models
from curve import predict_10_years

country = 'MEX'   # <<<<<<<<<<  Options = 'USA', 'CAN' or 'MEX'. Just modify that line

epochs = 130
# Loading data and configuring the LSTM
if country == 'USA':
    time_step = 25
    x_train, y_train, x_test, y_test, norm = TFP.load_data(path='TFP.csv', time_step=time_step, country='USA')  # USA data
    model = Models.model_usa(x_train)  # configuring the model
else:
    time_step = 5
    if country == 'CAN':
        x_train, y_train, x_test, y_test, _ = TFP.load_data(path='TFP.csv', time_step=time_step, country='CAN')  # CAN data
        model = Models.model_can(x_train)  # configuring the model
    elif country == 'MEX':
        x_train, y_train, x_test, y_test, _ = TFP.load_data(path='TFP.csv', time_step=time_step, country='MEX')  # MEX data
        model = Models.model_mex(x_train)  # configuring the model

# Training
history = model.fit(x_train, y_train, batch_size=8, epochs=epochs, verbose=2)

# Testing
predict = model.predict(x_test)

if country == 'USA':
    # Returning to the original values
    predict = norm.inverse_transform(predict)
    y_test = norm.inverse_transform(y_test)
    y_train = norm.inverse_transform(y_train)

# Curves
predict_10_years(y_train, y_test, predict, time_step)