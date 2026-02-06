"""
Script de prueba para el chatbot

Uso:
    python test_chatbot.py
"""
import os
from pathlib import Path
from chat_engine import ChatEngine


def main():
    """Función principal de prueba"""
    
    # Obtener API key de variable de entorno
    api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key:
        print("=" * 60)
        print("ERROR: No se encontró GROQ_API_KEY")
        print("=" * 60)
        print("\nPor favor:")
        print("1. Ve a https://console.groq.com")
        print("2. Crea una cuenta (gratis)")
        print("3. Genera una API key")
        print("4. Exporta la key:")
        print("   export GROQ_API_KEY='tu-api-key'")
        print("\nO ejecuta:")
        print("   GROQ_API_KEY='tu-key' python test_chatbot.py")
        return
    
    print("=" * 60)
    print("CHATBOT DE FINANZAS CUANTITATIVAS - MODO PRUEBA")
    print("=" * 60)
    
    # Ruta del proyecto (quant/)
    project_root = Path(__file__).parent.parent
    
    print(f"\nRuta del proyecto: {project_root}")
    print("\nOpciones:")
    print("1. Probar SIN RAG (solo conversación)")
    print("2. Probar CON RAG (indexa código y busca)")
    
    choice = input("\nElige (1/2) [2]: ").strip() or "2"
    
    enable_rag = choice == "2"
    
    print("\n" + "=" * 60)
    print("Inicializando chatbot...")
    print("=" * 60)
    
    try:
        engine = ChatEngine(
            api_key=api_key,
            project_root=str(project_root) if enable_rag else None,
            model="llama-3.3-70b-versatile",  # Modelo más reciente y potente
            enable_rag=enable_rag
        )
        
        print("\n✓ Chatbot inicializado correctamente")
        print(f"  Modelo: llama-3.3-70b-versatile (Llama 3.3 70B)")
        print(f"  RAG: {'Activado' if enable_rag else 'Desactivado'}")
        
    except Exception as e:
        print(f"\n✗ Error inicializando: {e}")
        print("\nPosibles soluciones:")
        print("- Verifica que GROQ_API_KEY sea válida")
        print("- Instala dependencias: pip install -r requirements.txt")
        return
    
    # Mensaje de bienvenida
    print("\n" + "=" * 60)
    print(engine.get_welcome_message())
    print("=" * 60)
    
    # Ejemplos de preguntas
    print("\nEjemplos de preguntas:")
    if enable_rag:
        examples = [
            "¿Cómo se calcula el Sharpe ratio en esta app?",
            "Muéstrame cómo se hace el cálculo del VaR",
            "¿Qué métodos usa el analizador CAPM?"
        ]
    else:
        examples = [
            "¿Qué es el Sharpe ratio?",
            "Explícame CAPM",
            "¿Qué diferencia hay entre VaR y Expected Shortfall?"
        ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")
    
    print("\nComandos especiales:")
    print("  /clear  - Limpiar memoria")
    print("  /history - Ver historial")
    print("  /exit - Salir")
    
    # Loop de conversación
    print("\n" + "=" * 60)
    
    while True:
        try:
            # Leer input del usuario
            user_input = input("\n💬 Tú: ").strip()
            
            if not user_input:
                continue
            
            # Comandos especiales
            if user_input.lower() == '/exit':
                print("\n👋 ¡Hasta luego!")
                break
            
            elif user_input.lower() == '/clear':
                engine.clear_memory()
                print("✓ Memoria limpiada")
                continue
            
            elif user_input.lower() == '/history':
                history = engine.get_history()
                print(f"\n📜 Historial ({len(history)} mensajes):")
                for msg in history[-10:]:  # Últimos 10
                    role_icon = "💬" if msg['role'] == 'user' else "🤖"
                    preview = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
                    print(f"  {role_icon} {preview}")
                continue
            
            # Generar respuesta
            print("\n🤖 Asistente: ", end="", flush=True)
            
            result = engine.respond(user_input)
            
            print(result['response'])
            
            # Mostrar fuentes si hay RAG
            if result.get('sources'):
                print("\n📚 Fuentes consultadas:")
                for source in result['sources'][:3]:  # Top 3
                    print(f"  • {source['file']} ({source['name']})")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrumpido. ¡Hasta luego!")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Por favor intenta de nuevo.")


if __name__ == "__main__":
    main()
