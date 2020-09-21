import numpy as np    
import matplotlib.pyplot as plt

def predict(x, w, b):
    y = np.dot(w.transpose(), x) + b
    print("<Y> calculada: ", y)
    
    if y >= 0:
        return 1
    else:
        return -1

def main():
    print("\n          Simulador de compuerta lógica AND y OR por medio del perceptrón\n")
    compuerta = int(input("Trabajar compuerta and (1) o compuerta OR (2): "))
    
    print("Ingrese los valores de los Pesos W\n")
    weights = []
    for i in range(2):
        x = float(input("W" + str(i+1) + ": "))
        weights.append(x)
    weights = np.array(weights)
    
    bias = float(input("Ingrese el valor del bias: "))
    
    print("\n Resultados: \n")
    
    x_Vector = np.array([[1, 1, -1, -1], [1, -1, 1, -1]])
    
    for i in range(4):
        print("<Y> funcion de activacion: ", predict(x_Vector[:,i], weights, bias), "\n")
        
    if compuerta == 1:
        plt.title("Compuerta AND", fontsize=20)
        plt.scatter(1, 1, s=100, color="green")
        plt.scatter(1, -1, s=100, color="red")
        plt.scatter(-1, 1, s=100, color="red")
        plt.scatter(-1, -1, s=100, color="red")
        plt.text(1,1, "1,1", fontsize=20)
        plt.text(1,-1, "1,-1", fontsize=20)
        plt.text(-1,1, "-1,1", fontsize=20)
        plt.text(-1,-1, "-1,-1", fontsize=20)
        
    elif compuerta == 2:
        plt.title("Compuerta OR", fontsize=20)
        plt.scatter(1, 1, s=100, color="green")
        plt.scatter(1, -1, s=100, color="green")
        plt.scatter(-1, 1, s=100, color="green")
        plt.scatter(-1, -1, s=100, color="red")
        plt.text(1,1, "1,1", fontsize=20)
        plt.text(1,-1, "1,-1", fontsize=20)
        plt.text(-1,1, "-1,1", fontsize=20)
        plt.text(-1,-1, "-1,-1", fontsize=20)
        
    x_values = [-3,3]
    y_values = [-(weights[0]/weights[1])*(-3) - (bias / weights[1]), 
                -(weights[0]/weights[1])*(3) - (bias / weights[1])]
    plt.plot(x_values, y_values)
    
if __name__ == "__main__":
    main()