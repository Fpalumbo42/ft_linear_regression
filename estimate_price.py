def read_theta():
    """Reads theta values from a file and prints them."""

    try:
        file = open("model.txt")
        thetas = file.read().split(',')
        theta = [float(t) for t in thetas]
        file.close()
        return theta
    except Exception as e:
        return [0,0]

def estimate_price(t0, t1, km):
    """Estimates the price of a car based on its mileage."""
    
    return t0 + (t1 * km)
    
    

def main():
    thetas = read_theta()
    
    # t0 is origin price, t1 decrease coefficient by km 
    t0 = thetas[0]
    t1 = thetas[1]
    
    
    km = float(input("Enter the mileage of the car in km: "))
   
    price = estimate_price(t0, t1, km)

    print(f"The estimated price of the car is: ${price:.2f}")

if __name__ == "__main__":
    main()