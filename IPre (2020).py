import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# IMPORTANTE: Añadir DOCSTRINGS, que expliquen lo que hace la función, parámetros, tipos de los parámetros, que retorna


# ------------------------------------------------------------
# Funciones base
# ------------------------------------------------------------

def calcular_c_i(capex_he, opex_he, r, n):
    """Calcula el costo de inversión (c_i)
    
    Argumentos:
        capex_he  (float) : Gasto capital del equipo de calor, medido en millones de dolares del año 2020 ($MM, 2020) 
        opex_he (float) : Gasto operacional anual del equipo de calor, medido en millones de dolares por año, del año 2020 ($MM/año, 2020)
        r (float) : Tasa de descuento anual, medido en porcentaje por año (%/año)
        n (int) : Vida útil del equipo de calor, medido en años (año)
    
    Devuelve:
        c_i (float) : Costo de inversión anual, medido en millones de dolares por año, del año 2020 ($MM, 2020)
    """
    factor_recuperacion = (r * (1 + r)**n) / ((1 + r)**n - 1)
    c_i = capex_he * factor_recuperacion + opex_he
    return c_i

def calcular_lcoh(c_i, q_delivered, u, c_fuel, n_t, ef_fuel, c_carbon): #($/kWh)
    """
    Calcula el costo nivelado del calor (LCOH)

    Argumentos:
        c_i (float) : Costo de inversión anual, medido en millones de dolares por año, del año 2020 ($MM, 2020)
        q_delivered (float) : Calor máximo entregado, medido en kWh por año (kWh/año)
        u (float) : Factor de utilización del equipo de calor, es decir, la cantidad de tiempo que el equipo está produciendo calor,
        dividio en su vida útil, no tiene unidad de medida (N.A.)
        c_fuel (float) : Costo de la fuente de energía, medido en dolar por kWh ($/kWh)
        n_t (float) : Eficiencia térmica de la entrega del calor, no tiene unidad de medida (N.A.)
        ef_fuel (float) : Factor de emisiones del ciclo de vida (EF) de la fuente de energía correspondiente, 
        medida en kg de CO2 por kWh (kg CO2/kWh)
        c_carbon (float) : Costo nivelado del carbono, medido en $ por kg de CO2 ($/kg CO2)

    Devuelve:
        lcoh (float) : Costo nivelado del calor en dolar por kWh ($/kWh)
    """
    lcoh = (c_i / (q_delivered * u)) + (c_fuel / n_t) + ((ef_fuel * c_carbon) / n_t)
    return lcoh

# ------------------------------------------------------------
# Datos base (Table 2 y 3, 2020)
# ------------------------------------------------------------
escenarios = {
    "Steam boiler 150°C": {
        "q_delivered": 5.13e7, # (kWh/año)
        "power_delivered": 5.86, # (MW_t)
        "u": 0.47,
        "r": 0.07,
        "tecnologias": {
            "NG":  {"capex_he": 0.21e6, "opex_he": 0.03*0.21e6, "c_fuel": 0.010, "ef_fuel": 0.24, "n_t": 0.80, "n": 15, "c_carbon": 0, "T": 150},
            "B-NG":{"capex_he": 0.21e6, "opex_he": 0.03*0.21e6, "c_fuel": 0.010, "ef_fuel": 0.24, "n_t": 0.80, "n": 15, "c_carbon": 0.058, "T": 150},
            "B-E": {"capex_he": 1.16e6, "opex_he": 0.01*1.16e6, "c_fuel": 0.067, "ef_fuel": 0.42, "n_t": 0.95, "n": 15, "c_carbon": 0.058, "T": 150},
            "B-H2":{"capex_he": 0.21e6, "opex_he": 0.03*0.21e6, "c_fuel": 0.079, "ef_fuel": 0.27, "n_t": 0.80, "n": 15, "c_carbon": 0.058, "T": 150},
            "G-H2":{"capex_he": 0.21e6, "opex_he": 0.03*0.21e6, "c_fuel": 0.175, "ef_fuel": 0.00, "n_t": 0.80, "n": 15, "c_carbon": 0, "T": 150},
            "ST":  {"capex_he": 2.11e6, "opex_he": (0.52+0.09)*1e6, "c_fuel": 0.0, "ef_fuel": 0.0, "n_t": 0.74, "n": 30, "c_carbon": 0, "T": 150},
            "B-HP":{"capex_he": 2.50e6, "opex_he": 0.04*2.50e6, "c_fuel": 0.067, "ef_fuel": 0.42, "n_t": 1.5, "n": 15, "c_carbon": 0.058, "T": 150},
        },
    },

    "Ethane cracker 850°C": {
        "q_delivered": 4.27e9, # Valor en el artículo era 4.27e9 (kWh/año)
        "power_delivered": 4.88, # (MW_t)
        "u": 1,
        "r": 0.07,
        "tecnologias": {
            "NG":  {"capex_he": 1.01573e9, "opex_he": 0.05*1.01573e9*(0.07*(1+0.07)**30)/((1+0.07)**30-1), "c_fuel": 0.010, "ef_fuel": 0.24, "n_t": 0.60, "n": 30, "c_carbon": 0, "T": 850},
            "B-NG":{"capex_he": 1.01573e9, "opex_he": 0.05*1.01573e9*(0.07*(1+0.07)**30)/((1+0.07)**30-1), "c_fuel": 0.010, "ef_fuel": 0.24, "n_t": 0.60, "n": 30, "c_carbon": 0.058, "T": 850},
            "B-E": {"capex_he": 1.52359e9, "opex_he": 0.05*1.52359e9*(0.07*(1+0.07)**30)/((1+0.07)**30-1), "c_fuel": 0.067, "ef_fuel": 0.42, "n_t": 0.71, "n": 30, "c_carbon": 0.058, "T": 850},
            "B-H2":{"capex_he": 1.01573e9, "opex_he": 0.05*1.01573e9*(0.07*(1+0.07)**30)/((1+0.07)**30-1), "c_fuel": 0.079, "ef_fuel": 0.27, "n_t": 0.60, "n": 30, "c_carbon": 0.058, "T": 850},
            "G-H2":{"capex_he": 1.01573e9, "opex_he": 0.05*1.01573e9*(0.07*(1+0.07)**30)/((1+0.07)**30-1), "c_fuel": 0.175, "ef_fuel": 0.00, "n_t": 0.60, "n": 30, "c_carbon": 0, "T": 850}, 
        },
    },

    "Glass melter 1600°C": {
        "q_delivered": 1.64e7, # (kWh/año)
        "power_delivered": 1.9, # (MW_t)
        "u": 1,
        "r": 0.07,
        "tecnologias": {
            "NG":  {"capex_he": 3.34e6, "opex_he": 0.03*3.34e6/12, "c_fuel": 0.010, "ef_fuel": 0.24, "n_t": 0.40, "n": 12, "c_carbon": 0, "T": 1600}, 
            "B-NG":{"capex_he": 3.34e6, "opex_he": 0.03*3.34e6/12, "c_fuel": 0.010, "ef_fuel": 0.24, "n_t": 0.40, "n": 12, "c_carbon": 0.058, "T": 1600},
            "B-E": {"capex_he": 3.80e6, "opex_he": 0.28*3.80e6*(0.07*(1+0.07)**12)/((1+0.07)**12-1), "c_fuel": 0.067, "ef_fuel": 0.42, "n_t": 0.80, "n": 12, "c_carbon": 0.058, "T": 1600}, 
            "B-H2":{"capex_he": 2.54e6, "opex_he": 0.03*2.54e6/12, "c_fuel": 0.079, "ef_fuel": 0.27, "n_t": 0.40, "n": 12, "c_carbon": 0.058, "T": 1600},
            "G-H2":{"capex_he": 2.54e6, "opex_he": 0.03*2.54e6/12, "c_fuel": 0.175, "ef_fuel": 0.00, "n_t": 0.40, "n": 12, "c_carbon": 0, "T": 1600}, 
        },
    }
}

"""
Este es un diccionario con todos los datos útiles del artículo. Este diccionario se usa bastantes veces en el código.
Tiene información respecto a cada parámetro relevante para cada fuente de energía, para cada escenario.
Cabe destacar que hay espacio para mejora, para los datos de opex_he. Ya que se entregan de diversas formas en el artículo, en algunos casos
se tuvo que ponderar para poder utilizar las funciones creadas.

También, no está de más aclarar que el parámetro 'c_carbon' ($/kg CO2), no se aplica para NG ni green, solo para los blue.
"""

# ------------------------------------------------------------
# Cálculo general y gráficos (con desglose de componentes)
# ------------------------------------------------------------

resultados_lcoh = {}
componentes_lcoh = {}

for nombre_esc, datos in escenarios.items():
    resultados_lcoh[nombre_esc] = {}
    componentes_lcoh[nombre_esc] = {}

    for tech, p in datos["tecnologias"].items():
        n = p["n"]
        r = datos["r"]
        u = datos["u"]
        capex_he = p["capex_he"]
        opex_he = p["opex_he"]
        c_fuel = p["c_fuel"]
        ef_fuel = p["ef_fuel"]
        n_t = p["n_t"]
        c_carbon = p["c_carbon"]

        """
        Se extrae la información del diccionario para su uso
        """

        # Factor de recuperación de capital
        factor_recuperacion = (r * (1 + r)**n) / ((1 + r)**n - 1)
        """
        El factor de recuperación (float) es la relación entre una anualidad constante y la anualidad en un tiempo en específico.
        Depende de: r (float) : Tasa de descuento anual, medido en porcentaje por año (%/año), 
        y n (int) : Vida útil del equipo de calor, medido en años (año). 
        """

        # Componentes de LCOH (Ecuación 1)
        capex_term = (capex_he * factor_recuperacion) / (datos["q_delivered"] * u)
        opex_term = opex_he / (datos["q_delivered"] * u)
        fuel_term = c_fuel / n_t
        carbon_term = (ef_fuel * c_carbon) / n_t

        """
        Vamos a dividir el LCOH según lo que aporta el capex, opex, combustible y carbono.
        """

        # LCOH total
        lcoh_total = capex_term + opex_term + fuel_term + carbon_term

        # Guardar resultados
        resultados_lcoh[nombre_esc][tech] = lcoh_total
        componentes_lcoh[nombre_esc][tech] = {
            "CAPEX": capex_term,
            "OPEX": opex_term,
            "FUEL": fuel_term,
            "CARBON": carbon_term
        }

    # ---------------------------
    # Gráfico por escenario
    # ---------------------------
    tecnologias = list(resultados_lcoh[nombre_esc].keys())
    capex_vals = [componentes_lcoh[nombre_esc][t]["CAPEX"] for t in tecnologias]
    opex_vals = [componentes_lcoh[nombre_esc][t]["OPEX"] for t in tecnologias]
    fuel_vals = [componentes_lcoh[nombre_esc][t]["FUEL"] for t in tecnologias]
    carbon_vals = [componentes_lcoh[nombre_esc][t]["CARBON"] for t in tecnologias]

    """
    Se grafica el aporte de cada aspecto mencionado anteriormente.
    """

    # Crear gráfico de barras apiladas
    plt.figure(figsize=(9, 5))

    plt.bar(tecnologias, capex_vals, label="CAPEX", color="#4575b4")
    plt.bar(tecnologias, opex_vals, bottom=capex_vals, label="OPEX", color="#91bfdb")
    plt.bar(tecnologias,
            fuel_vals,
            bottom=np.array(capex_vals) + np.array(opex_vals),
            label="FUEL",
            color="#fee090")
    plt.bar(tecnologias,
            carbon_vals,
            bottom=np.array(capex_vals) + np.array(opex_vals) + np.array(fuel_vals),
            label="CARBON",
            color="#fc8d59")

    # Etiquetas y formato
    plt.title(f"LCOH por Fuente - {nombre_esc}")
    plt.ylabel("LCOH [$/kWhₜ]")
    plt.xlabel("Fuente de energía")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()

    """
    Lo que viene son detalles para la visualización.
    """

    # Mostrar valores totales encima de cada barra
    totales = np.array(capex_vals) + np.array(opex_vals) + np.array(fuel_vals) + np.array(carbon_vals)
    for i, total in enumerate(totales):
        plt.text(i, total + 0.005, f"{total:.3f}", ha="center", va="bottom", fontsize=9)

    # 🔹 Escala fija del eje Y: de 0 a 0.5 con pasos de 0.1
    plt.ylim(0, 0.5)
    plt.yticks(np.arange(0, 0.51, 0.10))

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# Mostrar el diccionario final de resultados
# ------------------------------------------------------------
print("\n=== Resultados LCOH por escenario ===")
for esc, techs in resultados_lcoh.items():
    print(f"\n{esc}:")
    for t, v in techs.items():
        comps = componentes_lcoh[esc][t]
        print(f"  {t:6s}: {v:.4f} $/kWh_t  (CAPEX={comps['CAPEX']:.4f}, OPEX={comps['OPEX']:.4f}, FUEL={comps['FUEL']:.4f}, CARBON={comps['CARBON']:.4f})")

    """
    Se printean los aportes de cada aspecto en cada energía de cada escenario, así como el valor de los LCOH.
    """

#------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------
# Regresiones no lineales para eficiencia, CAPEX, OPEX, para cada fuente de energía.
# Ahora incluye impresión de:
# - Ecuación polinómica
# - Ecuación exponencial
# - Ecuación logarítmica
# ----------------------------------------------------------------------------------

# Variables a analizar
variables = ['n_t', 'capex_he', 'opex_he']
grado = 3 #  Se puede cambiar el grado si se quiere (usar 2 o 3 como recomendación)

# Detectar tecnologías
tecnologias = set()
for datos in escenarios.values():
    tecnologias.update(datos.get("tecnologias", {}).keys())

# =====================================================
# Procesar cada tecnología
# =====================================================
for tech in tecnologias:
    print(f"\n{'='*70}")
    print(f"📌 Fuente de energía: {tech}")
    print(f"{'='*70}")

    # Recolectar datos
    T_vals = []
    datos = {var: [] for var in variables}

    for nombre_escenario, info in escenarios.items():
        techs = info.get("tecnologias", {})
        if tech in techs:
            T = techs[tech].get("T")
            if T is None:
                continue
            T_vals.append(T)
            for var in variables:
                datos[var].append(techs[tech].get(var))

    # Si no hay datos suficientes, saltar (mín 2)
    if len(T_vals) < 2:
        print(f"⚠️ No hay suficientes datos para {tech}, se omite.")
        continue
    
    """
    Para hacer una regresión, se pone como requisito tener por lo menos 2 datos para cada fuente de energía, en caso que no
    se cumpla esto, no se realizará el proceso.
    """

    # Convertir a arrays numpy
    T_vals = np.array(T_vals).reshape(-1, 1)
    T_pred = np.linspace(min(T_vals), max(T_vals), 200).reshape(-1, 1)

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(f"Regresiones para tecnología: {tech}", fontsize=16)

    # =====================================================
    for row, var in enumerate(variables):

        print(f"\n\n🔧 Parámetro: {var}")
        print("-" * 70)

        y = np.array(datos[var])

        # ================================================
        # 1) REGRESIÓN POLINÓMICA
        # ================================================
        poly = PolynomialFeatures(degree=grado)
        T_poly = poly.fit_transform(T_vals)

        model_poly = LinearRegression()
        model_poly.fit(T_poly, y)

        y_pred_poly = model_poly.predict(poly.transform(T_pred))

        # Construir ecuación
        intercept = model_poly.intercept_
        coefs = model_poly.coef_

        eq_terms = [f"{intercept:.4e}"]
        for i, coef in enumerate(coefs[1:], start=1):
            eq_terms.append(f"{coef:.4e}·T^{i}")

        eq_poly = f"{var}(T) = " + " + ".join(eq_terms)

        print("\n📈 Ecuación polinómica:")
        print(eq_poly)

        # Gráfico
        ax = axes[row, 0]
        ax.scatter(T_vals, y)
        ax.plot(T_pred, y_pred_poly)
        ax.set_title(f"{var} — Polinómica (grado {grado})")
        ax.grid(True)

        # ================================================
        # 2) REGRESIÓN EXPONENCIAL: y = a·exp(bT)
        # ================================================
        print("\n📈 Ecuación exponencial:")

        try:
            mask_pos = y > 0
            T_pos = T_vals[mask_pos]
            y_pos = y[mask_pos]

            model_exp = LinearRegression()
            model_exp.fit(T_pos, np.log(y_pos))

            a = np.exp(model_exp.intercept_)
            b = model_exp.coef_[0]

            eq_exp = f"{var}(T) = {a:.4e} · exp({b:.4e}·T)"
            print(eq_exp)

            y_pred_exp = a * np.exp(b * T_pred)

            ax = axes[row, 1]
            ax.scatter(T_vals, y)
            ax.plot(T_pred, y_pred_exp)
            ax.set_title(f"{var} — Exponencial")
            ax.grid(True)

        except Exception as e:
            print("❌ No se pudo ajustar (requiere y>0)")

            ax = axes[row, 1]
            ax.text(0.3, 0.5, "No se pudo ajustar\n(exige y>0)")
            ax.set_title(f"{var} — Exponencial")
            ax.axis("off")

        # ================================================
        # 3) REGRESIÓN LOGARÍTMICA: y = a + b ln(T)
        # ================================================
        print("\n📈 Ecuación logarítmica:")

        try:
            mask_Tpos = T_vals.flatten() > 0
            T_log = np.log(T_vals[mask_Tpos])
            y_log = y[mask_Tpos]

            model_log = LinearRegression()
            model_log.fit(T_log.reshape(-1, 1), y_log)

            a_log = model_log.intercept_
            b_log = model_log.coef_[0]

            eq_log = f"{var}(T) = {a_log:.4e} + {b_log:.4e}·ln(T)"
            print(eq_log)

            y_pred_log = a_log + b_log * np.log(T_pred)

            ax = axes[row, 2]
            ax.scatter(T_vals, y)
            ax.plot(T_pred, y_pred_log)
            ax.set_title(f"{var} — Logarítmica")
            ax.grid(True)

        except Exception as e:
            print("❌ No se pudo ajustar (requiere T>0)")

            ax = axes[row, 2]
            ax.text(0.3, 0.5, "No se pudo ajustar\n(exige T>0)")
            ax.set_title(f"{var} — Logarítmica")
            ax.axis("off")

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------
# Gráficos capex_he/q_delivered v/s Temperatura y capex_he/power_delivered v/s Temperaturas
# -------------------------------------------------------------

# -------------------------------------------------------------
# Organizar datos por tecnología
# -------------------------------------------------------------
data_by_tech = {}

for escenario, datos in escenarios.items():
    q = datos["q_delivered"]
    power = datos["power_delivered"]

    for tech, tdata in datos["tecnologias"].items():
        capex = tdata["capex_he"]
        T = tdata["T"]

        if tech not in data_by_tech:
            data_by_tech[tech] = {
                "T": [],
                "capex_over_q": [],
                "capex_over_power": [],
                "escenario": []
            }

        data_by_tech[tech]["T"].append(T)
        data_by_tech[tech]["capex_over_q"].append(capex / q)
        data_by_tech[tech]["capex_over_power"].append(capex / power)
        data_by_tech[tech]["escenario"].append(escenario)

# -------------------------------------------------------------
# Crear dos gráficos por tecnología
# -------------------------------------------------------------
for tech, info in data_by_tech.items():

    T_vals = info["T"]
    cap_q = info["capex_over_q"]
    cap_p = info["capex_over_power"]
    escenarios_list = info["escenario"]

    # =============================
    # 1) Gráfico: capex_he / q_delivered vs T
    # =============================
    plt.figure(figsize=(8,5))
    plt.plot(T_vals, cap_q, marker="o")

    for i, esc in enumerate(escenarios_list):
        plt.annotate(esc, (T_vals[i], cap_q[i]), fontsize=8)

    plt.xlabel("Temperatura (°C)")
    plt.ylabel("Capex / Energía [USD*año/kWh]")
    plt.title(f"Tecnología: {tech} — Capex / Energía vs Temperatura")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # =============================
    # 2) Gráfico: capex_he / power_delivered vs T
    # =============================
    plt.figure(figsize=(8,5))
    plt.plot(T_vals, cap_p, marker="s")

    for i, esc in enumerate(escenarios_list):
        plt.annotate(esc, (T_vals[i], cap_p[i]), fontsize=8)

    plt.xlabel("Temperatura (°C)")
    plt.ylabel("Capex / Potencia [USD/MW]")
    plt.title(f"Tecnología: {tech} — Capex / Potencia vs Temperatura")
    plt.grid(True)
    plt.tight_layout()
    plt.show()