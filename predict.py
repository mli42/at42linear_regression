#!/usr/bin/env python3

import numpy as np
import utils

def main():
    theta = utils.get_theta()
    km, _ = utils.get_data()
    x = input("Enter an x value (float): ")

    try:
        x = float(x)
        if x < .0:
            raise Exception("negative mileage")
    except Exception as e:
        print(f"Getting x: {e}")
        return

    x = np.asarray([[x]])
    x = utils.minmax(x, np.min(km), np.max(km))
    prediction = utils.predict(x, theta)
    print(f"I'm predicting: {prediction[0][0]:.2f}")

if __name__ == "__main__":
    main()
