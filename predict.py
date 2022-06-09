import numpy as np
import utils

def main():
    theta = utils.get_theta()
    x = input("Enter an x value (float): ")

    try:
        x = float(x)
    except Exception as e:
        print(f"Getting x: {e}")
        return

    x = np.asarray([[x]])
    prediction = utils.predict(x, theta)
    print(f"I'm predicting: {prediction[0][0]}")

if __name__ == "__main__":
    main()
