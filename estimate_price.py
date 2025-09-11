def get_thetas(path):
    """get thetas values from a file"""

    try:
        file = open(path)
        thetas = file.read().split(',')
        theta = [float(t) for t in thetas]
        file.close()
        return theta[0], theta[1]
    except Exception as e:
        return [0,0]

def estimate_price(t0, t1, km):
    """Estimates the price of a car based on its mileage."""
    
    return t0 + (t1 * km)

def main():
    """Predict a price with the mileage of the car in km"""

    t0, t1 = get_thetas("model.txt")
    
    try:
        km = float(input("Enter the mileage of the car in km: "))
    except Exception as e:
        print(f"Please enter a valid number")
        exit()
   
    price = estimate_price(t0, t1, km)
    print(f"The estimated price of the car is: {price:.2f}")

if __name__ == "__main__":
    main()