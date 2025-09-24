from typing import List, Tuple

def generar_calendario_doble_vuelta(lista_equipos: List[str]) -> List[Tuple[str, str]]:
    """
    Genera un calendario de liga todos contra todos en dos vueltas (ida y vuelta)
    usando el método de rotación.
    """
    numero_equipos = len(lista_equipos)
    rotacion = lista_equipos[:]
    if numero_equipos % 2 != 0:
        rotacion.append("DESCANSO")
        numero_equipos += 1

    mitad = numero_equipos // 2
    jornadas: List[List[Tuple[str, str]]] = []

    for _ in range(numero_equipos - 1):
        partidos_jornada: List[Tuple[str, str]] = []
        for i in range(mitad):
            local = rotacion[i]
            visitante = rotacion[-(i+1)]
            if local != "DESCANSO" and visitante != "DESCANSO":
                partidos_jornada.append((local, visitante))
        jornadas.append(partidos_jornada)
        rotacion = [rotacion[0]] + [rotacion[-1]] + rotacion[1:-1]

    calendario: List[Tuple[str, str]] = []
    for partidos in jornadas:
        calendario.extend(partidos)  
    for partidos in jornadas:
        calendario.extend([(v, l) for l, v in partidos])  

    return calendario
