#!/usr/bin/env python3
"""
Script de test pour vérifier la connexion avec Ollama et le modèle Qwen 2.5:3b-instruct
"""

import requests
import json

def test_ollama_connection(base_url: str = "http://localhost:11434"):
    """Teste la connexion à Ollama"""
    print("=== Test de connexion Ollama ===\n")
    
    # 1. Vérifier que Ollama répond
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        print(f"✓ Ollama est accessible à {base_url}")
        models = response.json().get("models", [])
        print(f"  Modèles disponibles: {len(models)}")
        for model in models:
            print(f"    - {model.get('name', 'unknown')}")
    except requests.exceptions.RequestException as e:
        print(f"✗ Erreur de connexion à Ollama: {e}")
        print("\nAssurez-vous qu'Ollama est lancé avec: ollama serve")
        return False
    
    print()
    
    # 2. Vérifier que le modèle qwen2.5:3b-instruct est disponible
    model_name = "qwen2.5:3b-instruct"
    model_found = any(m.get("name", "").startswith("qwen2.5:3b") for m in models)
    
    if not model_found:
        print(f"✗ Le modèle {model_name} n'est pas disponible")
        print(f"\nPour l'installer, exécutez:")
        print(f"  ollama pull {model_name}")
        return False
    else:
        print(f"✓ Le modèle {model_name} est disponible")
    
    print()
    
    # 3. Test de génération simple
    print("=== Test de génération ===\n")
    test_prompt = "Réponds en une phrase: Qu'est-ce que la polyarthrite rhumatoïde?"
    
    try:
        payload = {
            "model": model_name,
            "prompt": test_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 200,
            }
        }
        
        print(f"Prompt: {test_prompt}")
        print("Génération en cours...\n")
        
        response = requests.post(f"{base_url}/api/generate", json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        answer = result.get("response", "").strip()
        
        print(f"✓ Réponse générée:")
        print(f"  {answer}")
        print(f"\n  Temps: {result.get('total_duration', 0) / 1e9:.2f}s")
        print(f"  Tokens: {result.get('eval_count', 0)}")
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Erreur lors de la génération: {e}")
        return False
    
    print()
    
    # 4. Test de l'API chat
    print("=== Test de l'API Chat ===\n")
    
    try:
        messages = [
            {"role": "system", "content": "Tu es un assistant médical."},
            {"role": "user", "content": "Cite 2 symptômes de la PR"}
        ]
        
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 150,
            }
        }
        
        print("Test de conversation...")
        response = requests.post(f"{base_url}/api/chat", json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        answer = result.get("message", {}).get("content", "").strip()
        
        print(f"✓ Réponse chat:")
        print(f"  {answer}\n")
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Erreur lors du chat: {e}")
        return False
    
    print("\n=== Tous les tests sont passés! ===")
    print(f"\nVous pouvez maintenant utiliser agent1_with_ollama.py")
    return True

if __name__ == "__main__":
    test_ollama_connection()