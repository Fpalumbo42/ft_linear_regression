from estimate_price import estimate_price
import math
import matplotlib.pyplot as plt
import numpy as np

def open_file(file_name):
    try:
        with open(file_name, 'r') as file:
            data = file.read()
        return data
    except Exception as e:
        print(f"Error opening file {file_name}: {e}")
        return None

def parse_data(data):
    try:
        lines = data.strip().split('\n')
        
        if not lines:
            raise ValueError("File is empty")
        
        if lines[0] != "km,price":
            raise ValueError("Invalid file format - expected header 'km,price'")
        
        km = []
        price = []
        
        for i, line in enumerate(lines[1:], 1):
            line = line.strip()
            
            if not line:
                continue
                
            parts = line.split(',')
            
            if len(parts) != 2:
                print(f"Warning: Line {i+1} has invalid format: '{line}' - skipping")
                continue
            
            try:
                km_val = float(parts[0])
                price_val = float(parts[1])
                km.append(km_val)
                price.append(price_val)
            except ValueError as e:
                print(f"Warning: Line {i+1} contains invalid numbers: '{line}' - skipping")
                continue
        
        if not km or not price:
            raise ValueError("No valid data found in file")
        
        m = len(km)
            
        return km, price, m
        
    except Exception as e:
        print(f"Error parsing data: {e}")
        return None, None
    
def save_model(t0, t1):
    try:
        with open("model.txt", "w") as f:
            f.write(f"{t0},{t1}")
    except Exception as e:
        print(f"Error saving model: {e}")

def standardize_features(km_list):
    
    # media
    mean_km = sum(km_list) / len(km_list)
    print(mean_km)
    variance = 0
    
    # ecart type
    for km in km_list:
        variance += (km - mean_km) ** 2
        print(f"le km {km} laaaaaaaaaaaaaaaa var {variance}" )
    gap_type = math.sqrt(variance / len(km_list))
    
    std_km = []
    for km in km_list:
        std_km.append((km - mean_km) / gap_type)
        
    return std_km, mean_km, gap_type

def denormalize_thetas(t0_std, t1_std, mean_km, gap_type):
    t0_final = t0_std - (t1_std * mean_km / gap_type)
    t1_final = t1_std / gap_type
    
    return t0_final, t1_final

def train_model(km_list = None, price_list = None, m = None, learning_rate = 0.1, max_iterations = 100):
    
    
    if km_list is None or price_list is None or m is None:
        print("Invalid data for train")
        return

    t0 = 0
    t1 = 0
    error_range = []
    
    km_list, mean_km, gap_type = standardize_features(km_list)
    
    for iter in range(max_iterations):
       error_range.clear()
       for i in range(m):
           result = estimate_price(t0, t1, km_list[i])
           error_range.append(result - price_list[i])
       
       tmp_t0 = learning_rate * (1/m) * sum(error_range)
       tmp_t1 = learning_rate * (1/m) * sum([error_range[i] * km_list[i] for i in range(m)])
       print(f"{tmp_t0} et {tmp_t1}")

       t0 -= tmp_t0
       t1 -= tmp_t1
    
    t0, t1 = denormalize_thetas(t0, t1, mean_km, gap_type)
       
    return t0, t1

def plot_data(km, price):

    plt.scatter(km, price, color='blue', label='Real data', alpha=0.7)
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price ')
    plt.title('Car Price vs Mileage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    data = open_file("data.csv")
    
    if data is None:
        print("Failed to read data.")
        return
    
    km, price, m = parse_data(data)
    
    if km is None or price is None:
        print("Failed to parse data.")
        return
        
    t0, t1 = train_model(km, price, m)
    save_model(t0, t1)
    plot_data(km, price)

if __name__ == "__main__":
    main()