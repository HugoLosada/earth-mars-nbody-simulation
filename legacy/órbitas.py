#Este código simula el movimiento de varios cuerpos celestes en el espacio, incluyendo la Tierra, Marte, el Sol y una nave espacial.
#VPython (vpython): Se utiliza para la visualización tridimensional de objetos y escenarios.
#Tkinter (tkinter): Se utiliza para crear una interfaz gráfica simple para obtener la entrada del usuario.
#La biblioteca de tiempo (time) en Python proporciona diversas funciones para trabajar con el tiempo y la fecha.
from bodies import Body
from vpython import *
import tkinter as tk
from tkinter import simpledialog
from math import sqrt, sin, cos, atan2
import time  

# Esta función utiliza Tkinter para mostrar un diálogo de entrada y solicitar al usuario información sobre el radio de la nave espacial (rship), su velocidad (vship), y el paso de tiempo (dt).
def get_user_input():
    root = tk.Tk()
    root.withdraw()

    rship = simpledialog.askfloat("Input", "Enter spacecraft radius:")
    vship = simpledialog.askfloat("Input", "Enter spacecraft velocity:")
    dt = simpledialog.askfloat("Input", "Enter time step:")

    return rship, vship, dt

# Constantes
G = 39.4784176
pi = 3.141592654
dospi = 2 * pi
pimedios = pi / 2

# parámetros iniciales de Marte.
rmarte = 1.53
mmarte = 3.214e-7
vmarte = sqrt(G / rmarte)
wmarte = vmarte / rmarte

# El usuario ingresa el radio de la nave espacial, su velocidad y el paso de tiempo.
rship, vship, dt = get_user_input()

a = (rship + rmarte) * 0.5
T = a**1.5
ANG = pi - wmarte * T / 2

# Posiciones iniciales
print('Initial Angular Position:', ANG * 180 / pi)

# Se calculan valores trigonométricos utilizados posteriormente.
sa = sin(ANG)
ca = cos(ANG)
sca = sin(pimedios + ANG)
cca = cos(pimedios + ANG)

# Calcular el radio mínimo de influencia
Rmin = rmarte * (mmarte)**0.4

# Se crean esferas representando el Sol, la Tierra, Marte y la nave espacial, con sus respectivos parámetros.
Sun = sphere(pos=vector(0, 0, 0), radius=0.05, color=color.yellow, make_trail=True, interval=10)
Sun.mass = 1
Sun.v = vector(0, 0, 0)

Earth = sphere(pos=vector(1, 0, 0), radius=0.03, color=color.blue, make_trail=True, interval=10)
Earth.mass = 3.004e-6
Earth.v = vector(0, 6.283, 0)

Mars = sphere(pos=vector(rmarte * ca, rmarte * sa, 0), radius=0.03, color=color.red, make_trail=True, interval=10)
Mars.mass = mmarte
Mars.v = vector(vmarte * cca, vmarte * sca, 0)

Ship = sphere(pos=vector(rship, 0, 0), radius=0.01, color=color.orange, make_trail=True, interval=10)
Ship.mass = 3.214e-28
Ship.v = vector(0, vship, 0)

# Se establece el valor de t como el paso de tiempo utilizado en la simulación.
dt = 2.73785078e-4

# Imprime en la consola el valor del radio de influencia mínimo de Marte (Rmin).
print('Mars Influence Radius:', Rmin)

# La variable flag se utiliza como indicador para el ajuste de trayectoria de la nave espacial cuando se acerca a Marte.
flag = 0

# Genera una marca de tiempo única para cada ejecución.
timestamp = time.strftime("%Y%m%d%H%M%S")

# Abre un nuevo archivo con la marca de tiempo en el nombre.
with open(f'posiciones_{timestamp}.txt', 'w') as file:
    #Se utiliza esta cadena para nombrar el archivo de salida. La parte {timestamp} se sustituye con la marca de tiempo única generada anteriormente.
    file.write("Simulación iniciada en {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))    
#strftime es una función que formatea la fecha y hora actuales según un formato especificado. En este caso, se utiliza el formato %Y%m%d%H%M%S, que representa el año, mes, día, hora, minuto y segundo. Esto crea una cadena que representa una marca de tiempo única.
    
    # Inicia un bucle infinito que representa el tiempo continuo en la simulación.
    while True:
        rate(100)    # Limita la velocidad de la simulación a 100 iteraciones por segundo.
        for body in [Earth, Mars, Ship]:
            if body == Ship:
                # Se calculan los vectores de posición de la nave espacial con respecto a Marte (DM) y al Sol (DS).
                DM = body.pos - Mars.pos
                DS = body.pos - Sun.pos

                # Se verifica si la distancia entre la nave espacial y Marte es mayor que el radio de influencia mínimo de Marte (Rmin).
                if mag(DM) > Rmin:
                    # Se calcula la aceleración debida a la gravedad del Sol, y se actualiza la posición y velocidad de la nave espacial utilizando la integración de Verlet.
                    as1 = -G * Sun.mass * DS / mag(DS)**3
                    body.pos += body.v * dt + 0.5 * as1 * dt**2
                    DS = body.pos - Sun.pos
                    as2 = -G * Sun.mass * DS / mag(DS)**3
                    body.v += 0.5 * (as1 + as2) * dt
                else:
                    if flag == 0:
                        print('Distance to Mars for Trajectory Adjustment:', mag(DM))
                        # Se aumenta temporalmente el radio de influencia de Marte.
                        Rmin *= 1
                        R = DM
                        Radio = mag(DM)
                        flag = 1

                        # Calcular nuevos componentes de velocidad para ajuste de trayectoria.
                        v = sqrt(G * Mars.mass / Radio)
                        w = v / Radio

                        # Calcular el ángulo polar para el vector de velocidad.
                        thita = atan2(R[1], R[0])

                        # Asegurar que θ (theta) esté en el rango correcto.
                        thita = thita % dospi

                        # Calcular componentes del vector de velocidad.
                        vnvector = vector(v * sin(thita + pi / 2), v * cos(thita + pi / 2), 0)
                        vtotal = vnvector + Mars.v

                        # Actualizar posición y velocidad de la nave espacial.
                        R = vector(Radio * cos(thita), Radio * sin(thita), 0)
                        body.pos = R + Mars.pos
                        thita += w * dt
            else:
                # Calcular la aceleración debido a la gravedad del Sol.
                distance = body.pos - Sun.pos
                a1 = -G * Sun.mass * distance / mag(distance)**3

                # Actualizar la posición usando la integración de Verlet.
                body.pos += body.v * dt + 0.5 * a1 * dt**2

                # Recalcular la aceleración después de la actualización de la posición.
                distance = body.pos - Sun.pos
                a2 = -G * Sun.mass * distance / mag(distance)**3

                # Actualizar la velocidad utilizando la integración de Verlet.
                body.v += 0.5 * (a1 + a2) * dt

        # Se escribe en el archivo posiciones.txt las posiciones actuales de la Tierra, Marte y la nave espacial, separadas por tabulaciones.
        file.write(f"{Earth.pos}\t{Mars.pos}\t{Ship.pos}\n")

