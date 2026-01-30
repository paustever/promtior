# Resumen del Proyecto – Chatbot RAG de Promtior

## Objetivo

El objetivo de este proyecto fue crear un chatbot que use la arquitectura RAG (Retrieval Augmented Generation) para responder preguntas sobre Promtior. La idea era que el bot solo use la información de los documentos y páginas web que yo le doy, sin inventarse nada. Quería que fuera confiable y claro.

---

## Tecnologías que usé

Para hacer esto usé varias herramientas:
- **Python 3.12**  
- **FastAPI**, para crear la API que recibe las preguntas y devuelve respuestas.  
- **LangChain**, para armar la cadena de recuperación y generación de respuestas (RAG).  
- **Ollama LLaMA 3**, para generar embeddings y las respuestas del modelo.  
- **Chroma**, para guardar los embeddings y buscar información relevante.  
- **PyPDFLoader y WebBaseLoader**, para cargar PDFs y páginas web oficiales.  
- **RecursiveCharacterTextSplitter**, para dividir los textos en pedacitos que el modelo pueda manejar.  
- **ChatPromptTemplate y RunnablePassthrough**, para organizar cómo el bot procesa las preguntas.

---

## Cómo lo armé

1. **Carga de documentos**  
   Primero cargué los PDFs y varias páginas web de Promtior. Los PDFs con `PyPDFLoader` y las webs con `WebBaseLoader`. Después los junté todos en una lista.  

2. **División en fragmentos**  
   Para que el modelo pudiera buscar la información, dividí los textos en pedazos de 1500 caracteres con un solapamiento de 300. Esto ayuda a que el bot tenga contexto suficiente para responder mejor.  

3. **Creación del vectorstore**  
   Cada fragmento lo transformé en embeddings usando Ollama, y los guardé en Chroma para poder buscarlos rápido.  
   - Probé varios valores de `chunk_size` y `k` para mejorar las respuestas, aunque no siempre quedaron perfectas.  

4. **Construcción del flujo RAG**  
   Armé la cadena para que el modelo busque primero en los documentos y luego genere la respuesta, con un prompt que le indica no inventar nada, responder solo con la información disponible y tratar sinónimos (“fundada” = “founded”).  

5. **API con FastAPI**  
   - Endpoint `/ask` para enviar preguntas y recibir respuestas.  
   - Endpoint `/` para verificar que la API funciona.  
   - Para no bloquear la API mientras se cargaban los documentos y embeddings, usé un thread que hace todo en segundo plano y avisa cuando el vectorstore está listo.

---

## Dificultades y cómo las solucioné

- **Respuestas inconsistentes**: a veces no entendía que “company” y “Promtior” eran lo mismo. Pues lograba contestar cuando fue promtior fundado pero no la compania.


- **Versiones de Ollama/LLaMA**: la versión que usé no siempre daba resultados perfectos. Por ahora documenté la versión lograda pero se que la misma puede mejorar.  

- **Tiempos de carga largos**: localmente funcionaba mejor que la API (daba respuestas mas exactas y mas completas)
  - Puse un aviso para que el usuario sepa si todavía no puede hacer preguntas, ya que sino se generaban bugs.

- **Ajustes de parámetros**: tuve que probar distintos valores de `chunk_size` y `k` para que el bot traiga información relevante. Todavía no es perfecto, pero funciona bastante bien.

---

## Qué aprendí

Aunque no quedó perfecto, me encantó hacerlo. Aprendí mucho durante el proceso (mire videos en youtube, lei docuementacion brindada y extra, ademas hice preguntas a ia) ya que es la primera vez que hago algo asi. Me gusta enfrentar problemas difíciles y probar soluciones, y este proyecto me demostró que puedo aprender rápido y resolver cosas técnicas complejas. Me motivó ver cómo, aunque con errores o inconsistencias, podía construir algo funcional. Me encantaria poder entrar al puesto para aprender en proyectos de IA generativa, porque realmente me interesa y disfruto resolver desafíos así.


## Despliegue
El proyecto está preparado para ser desplegado en cualquier nube, como AWS, Azure o Railway. Para pruebas locales, la API se ejecuta con Uvicorn y responde en el endpoint /ask. Se configuró la variable de entorno USER_AGENT=promtiortest para que las librerías de carga web funcionen correctamente. Aunque no se hizo un despliegue formal en la nube, el código está listo para subirlo a un servicio de hosting y funcionar sin modificaciones. La documentación oficial de LangChain sobre Langserve se puede usar como referencia si se desea un despliegue completo.
---

