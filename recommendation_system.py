# Sistema de recomendación
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Datos de ejemplo: usuarios y sus valoraciones de productos (0 = no valorado)
ratings = np.array([
    [5, 3, 0, 1],  # Usuario 1
    [4, 0, 4, 2],  # Usuario 2
    [1, 2, 5, 0]   # Usuario 3
])

# Calcular similitud entre usuarios
similarity_matrix = cosine_similarity(ratings)

# Generar recomendación para un usuario específico
def recomendar(user_id, n_recomendaciones=2):
    # Productos no valorados por el usuario
    productos_no_valorados = np.where(ratings[user_id] == 0)[0]
    
    # Predecir puntuaciones basadas en usuarios similares
    scores = []
    for producto in productos_no_valorados:
        usuarios_valoraron = ratings[:, producto] > 0
        similitud_relevante = similarity_matrix[user_id, usuarios_valoraron]
        valoraciones_relevantes = ratings[usuarios_valoraron, producto]
        
        if len(valoraciones_relevantes) > 0:
            score_predicho = np.dot(similitud_relevante, valoraciones_relevantes) / np.sum(similitud_relevante)
            scores.append((producto, score_predicho))
    
    # Ordenar por mejor puntuación
    scores.sort(key=lambda x: x[1], reverse=True)
    return [producto for producto, _ in scores[:n_recomendaciones]]

# Ejemplo de uso
usuario_objetivo = 0
recomendaciones = recomendar(usuario_objetivo)
print(f"Recomendaciones para Usuario {usuario_objetivo+1}: Productos {recomendaciones}")