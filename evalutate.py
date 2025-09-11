from train import parse_data, open_file
from estimate_price import estimate_price, get_thetas
import math

def get_mean_squared_error(price, computed_results, m):
    se = 0
    for i, value in enumerate(price):
        se += (value - computed_results[i]) ** 2
    mse = 1/m * se
    return mse

def get_mean_absolute_percentage_error(price, computed_results, m):
    gap = 0
    for i in range(len(price)):
        gap += abs(price[i] - computed_results[i]) / price[i]
    return 1/m * gap * 100

def write_evaluation_report(mse, rmse, mape, t0, t1, m, filename="evaluation_results.txt"):
    try:
        with open(filename, 'w') as f:
            f.write("LINEAR REGRESSION MODEL EVALUATION REPORT\n")
            f.write(f"Price decreases by {abs(t1):.4f}€ for each additional kilometer\n")
            f.write("ACCURACY METRICS:\n")
            f.write("-" * 17 + "\n")
            f.write(f"Mean Squared Error (MSE): {mse:.2f}\n") 
            f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}€\n")
            f.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%\n")
        
    except Exception as e:
        print(f"Error writing evaluation report: {e}")

def main():
    computed_results = []
    
    data = open_file("data.csv")
    
    if data is None:
        print("Failed to read data.")
        return
    
    km, price, m = parse_data(data)
    t0, t1 = get_thetas("model.txt")
    
    for value in km:
        computed_results.append(estimate_price(t0, t1, value))
    
    mse = get_mean_squared_error(price, computed_results, m)
    mape = get_mean_absolute_percentage_error(price, computed_results, m)
    rmse = math.sqrt(mse)
    
    # Write detailed report instead of printing
    write_evaluation_report(mse, rmse, mape, t0, t1, m)
    
    print(f"RMSE: {rmse:.2f}€ | MAPE: {mape:.2f}%")

if __name__ == "__main__":
    main()