# Linear Regression Project - Technical Deep Dive

## Project Overview

This project implements a **simple linear regression model** to predict car prices based on mileage using the **gradient descent algorithm**. The goal is to understand fundamental machine learning concepts through hands-on implementation.

**Core Concept**: Find the optimal linear relationship `price = θ₀ + θ₁ × mileage` that best fits our training data.

---

## Mathematical Foundation

### The Linear Hypothesis

Our model uses the linear hypothesis:

```
estimatePrice(mileage) = θ₀ + θ₁ × mileage
```

Where:
- **θ₀ (theta0)**: The y-intercept (base price when mileage = 0)
- **θ₁ (theta1)**: The slope (price change per unit of mileage)
- **mileage**: The input feature (independent variable)
- **estimatePrice**: The predicted output (dependent variable)

### Cost Function (Implicit)

Although not explicitly implemented, our algorithm minimizes the **Mean Squared Error**:

```
J(θ₀, θ₁) = (1/2m) × Σ(estimatePrice(mileage[i]) - price[i])²
```

Where:
- **m**: Number of training examples (24 cars in our dataset)
- **Σ**: Summation over all training examples
- **price[i]**: Actual price of car i
- **estimatePrice(mileage[i])**: Predicted price of car i

---

## Gradient Descent Algorithm

### Core Principle

Gradient descent is an iterative optimization algorithm that finds the minimum of our cost function by:
1. Computing the gradient (partial derivatives) of the cost function
2. Moving in the opposite direction of the gradient
3. Repeating until convergence

### Update Formulas

The project uses these specific gradient descent update rules:

```
tmpθ₀ = learningRate × (1/m) × Σ(estimatePrice(mileage[i]) - price[i])

tmpθ₁ = learningRate × (1/m) × Σ((estimatePrice(mileage[i]) - price[i]) × mileage[i])
```

Then simultaneously update:
```
θ₀ = θ₀ - tmpθ₀
θ₁ = θ₁ - tmpθ₁
```

### Mathematical Intuition

**For θ₀ (bias term):**
- If predictions are consistently too high → positive error → decrease θ₀
- If predictions are consistently too low → negative error → increase θ₀
- The update is the **average error** across all examples

**For θ₁ (slope term):**
- The update considers both the error AND the mileage value
- High-mileage cars with large errors have more influence on slope adjustment
- This is because θ₁ represents "price change per mile"

### Simultaneous Update

**Critical**: Both θ₀ and θ₁ must be updated simultaneously using the same iteration's values:

```python
# CORRECT
tmp_theta0 = calculate_theta0_update()
tmp_theta1 = calculate_theta1_update()
theta0 = theta0 - tmp_theta0
theta1 = theta1 - tmp_theta1

# WRONG - sequential update
theta0 = theta0 - calculate_theta0_update()
theta1 = theta1 - calculate_theta1_update()  # Uses updated theta0!
```

---

## Feature Normalization (Standardization)

### The Problem: Scale Mismatch

Our raw dataset has features on vastly different scales:
- **Mileage**: 22,899 - 240,000 (large numbers)
- **Price**: 3,650 - 8,290 (smaller numbers)

This creates numerical instability in gradient descent:

```python
# Without normalization
error = -1000  # euros
mileage = 150000  # km
gradient_contribution = error × mileage = -150,000,000  # Too big
```

### Standardization Formula

We apply **Z-score standardization**:

```
x_standardized = (x - μ) / σ
```

Where:
- **μ (mu)**: Mean of the feature
- **σ (sigma)**: Standard deviation of the feature

### Implementation Details

```python
def standardize_features(km_list):
    # Calculate mean
    mean_km = sum(km_list) / len(km_list)
    
    # Calculate standard deviation
    variance = sum([(km - mean_km)**2 for km in km_list]) / len(km_list)
    std_km = sqrt(variance)
    
    # Standardize each value
    standardized_km = [(km - mean_km) / std_km for km in km_list]
    
    return standardized_km, mean_km, std_km
```

### After Standardization

- **Original mileage**: [240000, 139800, 150500, ...]
- **Standardized mileage**: [2.19, 0.46, 0.65, ...]
- **Properties**: Mean ≈ 0, Standard deviation ≈ 1

---

## Theta Denormalization

### The Challenge

After training on standardized data, our θ values work only with standardized inputs:

```
price = θ₀_std + θ₁_std × mileage_standardized
```

But users want to input raw mileage values!

### Mathematical Derivation

Starting with the standardized equation:
```
price = θ₀_std + θ₁_std × (mileage - μ) / σ
```

Expanding:
```
price = θ₀_std + θ₁_std × mileage / σ - θ₁_std × μ / σ
```

Rearranging:
```
price = (θ₀_std - θ₁_std × μ / σ) + (θ₁_std / σ) × mileage
```

Therefore:
```
θ₀_final = θ₀_std - (θ₁_std × μ / σ)
θ₁_final = θ₁_std / σ
```

### Implementation

```python
def denormalize_thetas(theta0_std, theta1_std, mean_km, std_km):
    theta0_final = theta0_std - (theta1_std * mean_km / std_km)
    theta1_final = theta1_std / std_km
    return theta0_final, theta1_final
```

---

## Learning Rate and Convergence

### Learning Rate Impact

The learning rate (α) controls the step size in gradient descent:

- **Too large**: Algorithm overshoots → divergence → NaN values
- **Too small**: Algorithm converges very slowly
- **Just right**: Smooth convergence to optimal solution

### Typical Values

For normalized data:
- **Good starting point**: 0.01
- **Conservative**: 0.001
- **Aggressive**: 0.1 (risk of divergence)

### Convergence Behavior

**Iteration 1**: θ₀ = 0, θ₁ = 0 → All predictions = 0 (terrible)
**Iteration 10**: θ₀ = 1500, θ₁ = -0.02 → Better predictions
**Iteration 100**: θ₀ = 8500, θ₁ = -0.025 → Near-optimal

---

## Training Process Flow

### Step-by-Step Execution

1. **Initialize**: θ₀ = 0, θ₁ = 0
2. **Normalize features**: Convert mileage to standardized values
3. **For each iteration**:
   - Calculate predictions for all training examples
   - Compute errors: prediction - actual_price
   - Apply gradient descent formulas
   - Update θ₀ and θ₁ simultaneously
4. **Denormalize**: Convert θ values back to work with raw data
5. **Save model**: Store final θ₀ and θ₁ for prediction program

### Implementation in Code

```python
def train_model(km_list, price_list, m, learning_rate=0.01, max_iterations=100):
    # Initialize parameters
    theta0, theta1 = 0, 0
    
    # Normalize features
    km_std, mean_km, std_km = standardize_features(km_list)
    
    # Gradient descent loop
    for iteration in range(max_iterations):
        errors = []
        
        # Calculate prediction errors for all examples
        for i in range(m):
            prediction = theta0 + theta1 * km_std[i]
            error = prediction - price_list[i]
            errors.append(error)
        
        # Apply the assignment formulas
        tmp_theta0 = learning_rate * sum(errors) / m
        tmp_theta1 = learning_rate * sum([errors[i] * km_std[i] for i in range(m)]) / m
        
        # Update parameters (note the subtraction!)
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1
    
    # Denormalize for final use
    theta0_final, theta1_final = denormalize_thetas(theta0, theta1, mean_km, std_km)
    
    return theta0_final, theta1_final
```

**Key Implementation Notes:**
- We subtract the tmp_theta values because we want to move toward lower error
- The formulas give us the direction and magnitude of the adjustment
- Parameters are updated simultaneously using the same iteration's values

---

## Model Interpretation

### Final Parameters

After training, you might get:
- **θ₀ ≈ 8500**: Base price of a car with 0 mileage
- **θ₁ ≈ -0.021**: Price decreases by 2.1 cents per kilometer

### Prediction Examples

```python
# High mileage car
price = 8500 + (-0.021) × 200000 = 4300€

# Low mileage car  
price = 8500 + (-0.021) × 50000 = 7450€
```

### Model Limitations

This simple linear model only considers mileage. Real-world car pricing depends on:
- Brand and model
- Year and condition
- Features and equipment
- Market location
- Maintenance history

---

## Why This Approach Works

### Statistical Learning

The algorithm **discovers** the hidden relationship between mileage and price by:
1. **Pattern recognition**: Finding the line that best fits the data
2. **Error minimization**: Reducing prediction mistakes iteratively
3. **Generalization**: Learning from examples to predict unseen data

### Gradient Descent Intuition

Think of gradient descent as **rolling a ball down a hill**:
- The hill represents our cost function
- The ball's position represents our current θ values
- Gradient descent finds the bottom of the hill (minimum cost)
- Each iteration moves the ball closer to the bottom

### Linear Regression Assumptions

Our model assumes:
1. **Linear relationship**: Price changes linearly with mileage
2. **Independence**: Each car sale is independent
3. **Homoscedasticity**: Error variance is constant across all mileage values
4. **No outliers**: Extreme values don't dominate the model

---

## Conclusion

This project demonstrates fundamental machine learning concepts:
- **Supervised learning**: Learning from labeled examples
- **Linear models**: Simple but powerful for understanding relationships
- **Optimization**: Finding the best parameters through gradient descent
- **Feature engineering**: The importance of data preprocessing
- **Model evaluation**: Understanding what makes a good prediction