import math
import numpy as np
import matplotlib.pyplot as plt


def f21(x):
    return 4.0**x - 5.0*x - 2.0

def df21(x):
    return (4.0**x) * math.log(4.0) - 5.0

def phi21(x):
    return math.log(5.0*x + 2.0, 4.0)

def solve21():
    print(" ЗАДАНИЕ 2.1. Уравнение: 4^x - 5x - 2 = 0")
    print(" (Корень на отрезке [1, 2], x0 = 1.7)")
    
    eps = 1e-6
    x0 = 1.7
    max_iter = 50

    print("\n Метод простой итерации ")
    print(f"{'Итер':<6} | {'x':<15} | {'|f(x)|':<15} | {'Погрешность':<15}")
    x = x0
    for it in range(1, max_iter + 1):
        x_next = phi21(x)
        err = abs(x_next - x)
        print(f"{it:<6} | {x_next:<15.8f} | {abs(f21(x_next)):<15.8f} | {err:<15.8f}")
        x = x_next
        if err < eps:
            break

    print("\n Метод Ньютона ")
    print(f"{'Итер':<6} | {'x':<15} | {'|f(x)|':<15} | {'Погрешность':<15}")
    x = x0
    for it in range(1, max_iter + 1):
        x_next = x - f21(x) / df21(x)
        err = abs(x_next - x)
        print(f"{it:<6} | {x_next:<15.8f} | {abs(f21(x_next)):<15.8f} | {err:<15.8f}")
        x = x_next
        if err < eps:
            break
            
    print(f"\nОтвет 2.1: x = {x:.8f}")
    return x



def solve22():
    A = 3.0
    print("\n")
    print(" ЗАДАНИЕ 2.2. Система:")
    print(" 3*x1 - cos(x2) = 0")
    print(" 3*x2 - exp(x1) = 0")
    print(" (Начальное приближение x1=0.3, x2=0.45)")


    eps = 1e-6
    x1, x2 = 0.3, 0.45
    max_iter = 50


    print("\n Метод простой итерации ")
    print(f"{'Итер':<6} | {'x1':<12} | {'x2':<12} | {'F1':<12} | {'F2':<12} | {'Погрешность':<12}")
    x1, x2 = 0.3, 0.45
    for it in range(1, max_iter + 1):
        nx1 = math.cos(x2) / A
        nx2 = math.exp(x1) / A
        err = math.sqrt((nx1 - x1)**2 + (nx2 - x2)**2)
        
        x1, x2 = nx1, nx2
        f1 = A*x1 - math.cos(x2)
        f2 = A*x2 - math.exp(x1)
        
        print(f"{it:<6} | {x1:<12.8f} | {x2:<12.8f} | {abs(f1):<12.8f} | {abs(f2):<12.8f} | {err:<12.8f}")
        if err < eps:
            break

    print("\n Метод Ньютона ")
    print(f"{'Итер':<6} | {'x1':<12} | {'x2':<12} | {'F1':<12} | {'F2':<12} | {'Погрешность':<12}")
    x1, x2 = 0.3, 0.45
    for it in range(1, max_iter + 1):
        f1 = A*x1 - math.cos(x2)
        f2 = A*x2 - math.exp(x1)
        
        J11 = A
        J12 = math.sin(x2)
        J21 = -math.exp(x1)
        J22 = A
        
        det = J11*J22 - J12*J21
        
        dx1 = (-f1*J22 + f2*J12) / det
        dx2 = (-f2*J11 + f1*J21) / det
        
        x1 += dx1
        x2 += dx2
        
        err = math.sqrt(dx1**2 + dx2**2)
        
        f1_new = A*x1 - math.cos(x2)
        f2_new = A*x2 - math.exp(x1)
        
        print(f"{it:<6} | {x1:<12.8f} | {x2:<12.8f} | {abs(f1_new):<12.8f} | {abs(f2_new):<12.8f} | {err:<12.8f}")
        if err < eps:
            break
            
    print(f"\nОтвет 2.2: x1 = {x1:.8f}, x2 = {x2:.8f}")
    return x1, x2

def plot_graphs(root21, root22_x1, root22_x2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.canvas.manager.set_window_title('Графическое определение нач. приближения (Вариант 17)')

    x_vals = np.linspace(-0.5, 2.5, 400)
    y_vals = 4.0**x_vals - 5.0*x_vals - 2.0
    
    ax1.plot(x_vals, y_vals, 'g-', label='y = 4^x - 5x - 2')
    ax1.axhline(0, color='black', linewidth=1) # Ось X
    ax1.axvline(0, color='black', linewidth=1) # Ось Y
    ax1.plot(root21, 0, 'ro', markersize=8, label=f'Корень x={root21:.4f}')
    
    ax1.set_title('Задание 2.1: 4^x - 5x - 2 = 0')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    x1_vals = np.linspace(-0.33, 0.33, 400)
    
    mask = (3*x1_vals >= -1) & (3*x1_vals <= 1)
    x1_vals = x1_vals[mask]
    
    y1_vals = np.arccos(3 * x1_vals)      
    y2_vals = np.exp(x1_vals) / 3.0       
    
    ax2.plot(x1_vals, y1_vals, 'g-', label='x2 = arccos(3*x1)')
    ax2.plot(x1_vals, y2_vals, 'b-', label='x2 = exp(x1) / 3')
    ax2.plot(root22_x1, root22_x2, 'ro', markersize=8, label=f'Корень\n({root22_x1:.3f}, {root22_x2:.3f})')
    
    ax2.set_title('Задание 2.2: Система уравнений')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    root_21 = solve21()
    
    root_22_x1, root_22_x2 = solve22()
    
    print("\n[Инфо] Открытие окна с графиками...")
    plot_graphs(root_21, root_22_x1, root_22_x2)