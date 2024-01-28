# README

## Estructura del proyecto 

###	Backend: Código python 
- **Config**: Carpeta de configuración de parámetros para los scripts:
    - **Model_parameters/**: 
        - **common_parameters/** :
            - **common_parameters.json**: Algunas variables comunes que tienen que ver con el nombre de las carpetas que se manejan para el almacenamiento interno de los modelos 
- **metadata/**: algunos json de configuración que listan parámetros específicos para cada modelo así como sus descripciones, y nombres legibles para poder usar luego en la parte del frontend
-	***.json:** Un json para cada uno de los modelos donde el * representa el nombre del modelo según la librería sklearn. Esto es para que se quede guardado en memoria para cada usuario según cambia las configuraciones del modelo
- **Data**: carpeta no trackeada en el git. Sirve para guardar todos los dataset que se generar a partir del código fuente principal y poder usarlo posteriormente si fuera necesario. 
- **ESRNN**: Carpeta con el código de la librería esrnn de Python (modelo ganador en el M4 competition) modificada para compatibilizar versiones (los autores no han mantenido el código actualizado así que ha sido necesario modificarlo)
- **global_creds**: No trackeado en git. Carpeta donde se guardan credenciales necesario para el acceso a ciertos servicios: credenciales de acceso a bbdd del servidor, etc.
- **Models**: Carpeta donde se guardan los modelos que se ejecutan en el código fuente principal. 
- **src/** : carpeta del código fuente.
    - **cleaning.py**:   Archivo de funciones donde se guardan el código relacionado con el preprocesamiento de los datos. Cada función está documentada detalladamente con descripciones, parámetros, valores de retorno y ejemplos.
        - prepare_dataframe_from_db: Esta función toma un DataFrame que representa datos de una base de datos y realiza varios pasos de preprocesamiento. Elimina la columna 'id_data', filtra las filas según los identificadores de variables especificados en 'cols_for_query', y opcionalmente maneja la columna 'timestamp' convirtiéndola a formato datetime y ajustando a UTC si se requiere.
        - handle_outliers: Es un marcador de posición para la lógica de manejo de valores atípicos, que se debe implementar según las necesidades específicas de los datos.
        - resample_data: Reorganiza los datos de series temporales en un DataFrame a una frecuencia uniforme. Agrupa los datos por 'id_device', 'id_sensor' y 'id_variable', y aplica una función de agregación especificada para calcular los valores remuestreados.
        - interpolate_data: Rellena los valores faltantes en los datos de series temporales usando un método de interpolación especificado. La función espera que el DataFrame tenga una columna 'timestamp'.
        - process_time_series_data: Función envoltorio que llama a resample_data e interpolate_data secuencialmente. También incluye la opción de manejar valores atípicos si se proporcionan las columnas relevantes.
        - Métodos personalizados de interpolación, por ejemplo, interpolate_mv_local_median: Realiza la interpolación de valores faltantes en un conjunto de datos utilizando la mediana local. Esta función utiliza un enfoque basado en la mediana local para interpolar valores faltantes en un arreglo 2D, aumentando iterativamente la ventana de búsqueda hasta encontrar una mediana no NaN para cada valor faltante.
    - **eda.py**: Archivo de funciones donde se guarda el código con la exploración de los datos y métricas estadísticas para el resumen de los mismos 
    - **predictions.py**: Este script de Python implementa un flujo completo para el procesamiento y predicción de series temporales utilizando modelos de aprendizaje automático. Incluye varias funciones clave:
        - Funciones de preprocesamiento relacionadas con el bloque de algoritmos de machine learning: Incluyen métodos para preparar datos de una base de datos, procesar series temporales (resampleo e interpolación), y generar características retardadas (lags) y adelantadas (leads).
        - Funciones de Modelado y Predicción: Permiten crear y entrenar modelos de aprendizaje automático, predecir con estos modelos y procesar los resultados de las predicciones. Incluyen la habilidad de escalar datos y revertir esta escala en las predicciones.
        - Funciones de Evaluación: Proporcionan herramientas para evaluar el rendimiento del modelo (por ejemplo, calculando el error absoluto medio) y resumir los resultados.
        - Funciones Auxiliares y de Manejo de Datos: Incluyen métodos para cargar y guardar configuraciones, modelos, datos preprocesados y resultados. También se encargan de la gestión de logs y la importación de datos.
        - Flujo Principal: run_time_series_prediction_pipeline es la función principal que une todo el flujo de trabajo, desde la carga de configuraciones y datos, hasta la formación del modelo, la realización de predicciones y la evaluación de resultados.
        - Visualisations.py: Funciones relacionadas con la visualización de datos. Se utilizan fundamentalmente gráficos con la librería plotly porque permite la creación de gráficos dinámicos que luego se pueden incrutrar fácilmente en el backend 
    - **ConfigManager.py*: Este script en Python define una clase ConfigManager que maneja la carga, guardado y actualización de archivos de configuración en formato JSON. Las funcionalidades clave son:
        - Inicialización: Establece el camino al directorio que contiene los archivos de configuración.
        - Carga de Configuraciones: Permite cargar archivos de configuración específicos desde un subdirectorio opcional.
        - Guardado de Configuraciones: Facilita el guardado de configuraciones en archivos JSON, con la opción de crear subdirectorios si no existen.
        - Listado de Configuraciones: Enumera todos los archivos de configuración disponibles en un subdirectorio específico.
        - Actualización de Configuraciones: Actualiza claves específicas en un archivo de configuración y en todos los archivos de un subdirectorio. Realiza conversiones de tipo de datos de manera segura para ajustarse al tipo original del valor de configuración.
    - **PersistenceManage.py*: El script proporciona una función llamada persist_model_to_disk_structure para guardar estructuras de modelos de aprendizaje automático y sus objetos asociados en el disco. Esta función utiliza una clase llamada PersistenceManager para manejar las operaciones de archivo. Las características clave incluyen:
        - Guardar Modelo, Metadatos y Escalador: La función guarda un objeto de modelo, metadatos y un escalador en una estructura de directorios específica. La ruta se construye utilizando parámetros como el nombre del modelo, el rango de entrenamiento y el tiempo de ejecución.
        - Guardar Datos Preprocesados: Opcionalmente, guarda datos preprocesados en un directorio especificado. Esto es útil cuando los datos deben ser procesados antes de ser utilizados por el modelo.
        - Crear Banderas: Crea archivos de bandera, como training-done.txt, para indicar la finalización de ciertos procesos, como el entrenamiento del modelo.
        - Guardar Objetos Adicionales: Permite la opción de guardar objetos adicionales definidos por el usuario, que pueden ser necesarios para el modelo o el proceso de análisis de datos.
        - Manejo de Errores: La función incluye manejo de errores para asegurar que cualquier problema durante el proceso de persistencia sea capturado y registrado, facilitando la depuración y asegurando la integridad de los datos guardados.
    - **OwnLog.py**: Clase para gestionar un log personalizado de todo el pipeline de los datos
    - **own_utils.py**: Funciones útiles usadas en distintas partes del proyecto 
    - **sql_utils.py**: Funciones útiles relacionadas con el manejo de base de datos en mysql
    - **Otros archivos .py**: relacionados con inicializaciones o pipelines de ejecuciones que agrupan el orden de ejecuciones de las funciones anteriores. Esto esta hecho para que el proyecto pueda ejecutarse en .py. Sin embargo, la plataforma no está gestionada por código si no por un frontend que permite manejar las funciones anteriormente descritas mediante una interface amigable y, en la propia interacción del usuario con la interface se genera el pipeline de las funciones anteriormente descritas adecuadamente 
    - Notebooks para la exploración de los datos: No están trackeados y se están usando para pasar el código a limpio 
###	Frontend: 

Framework django. El framework Django es una herramienta de desarrollo web de alto nivel en Python que fomenta un diseño limpio y pragmático. Es conocido por su simplicidad, flexibilidad y robustez, lo que lo convierte en una opción popular para el desarrollo de aplicaciones web. Aquí hay algunos aspectos clave de Django que se generan y se utilizan al emplear este framework:
- Modelo-Vista-Controlador (MVC): Django sigue el patrón de diseño MVC, que separa la aplicación en tres componentes interconectados. Estos son:
    - Modelos: Definen la estructura de la base de datos. Django utiliza una ORM (Object-Relational Mapping) para mapear los objetos de los modelos a las tablas de la base de datos, facilitando la manipulación de los datos.
    - Vistas: Encargadas de la lógica de la aplicación y la interacción con los modelos para procesar y responder a las solicitudes de los usuarios.
    - Controladores: En Django, la funcionalidad del controlador está incorporada en las vistas y el enrutamiento de URLs, que dirigen las solicitudes del usuario a las vistas apropiadas.
- Enrutamiento de URL: Django permite diseñar URLs legibles y fáciles de usar. El sistema de enrutamiento de URLs dirige las solicitudes de los usuarios a las vistas correctas basadas en la URL.
- Plantillas: El sistema de plantillas de Django es potente y extensible, permitiendo crear interfaces de usuario dinámicas. Las plantillas separan la presentación de la lógica de la aplicación, facilitando la mantenimiento y escalabilidad.
- Panel de Administración: Django proporciona un panel de administración automático y personalizable que permite a los usuarios gestionar el contenido de la aplicación, como modelos y usuarios, de manera intuitiva.
- Seguridad: Django incluye características de seguridad incorporadas como protección contra ataques CSRF (Cross Site Request Forgery), SQL injection, y XSS (Cross Site Scripting). Maneja la autenticación y autorización, proporcionando una manera segura de manejar cuentas de usuario y permisos.
- ORM y Migraciones de Base de Datos: Con Django, puedes definir tus modelos de base de datos en Python y realizar cambios en la estructura de la base de datos a través de migraciones. Esto abstrae la mayoría de las tareas relacionadas con SQL y facilita el mantenimiento de la base de datos.
- Aplicaciones Reutilizables: Django fomenta el desarrollo de aplicaciones reutilizables, que pueden ser "enchufadas" en cualquier proyecto de Django. Esto promueve la reutilización del código y una arquitectura modular.

En el contexto de la plataforma se han creado dos aplicaciones. Una es la de model_manager que es donde se gestiona principalmente toda la plataforma y donde están situadas todas las templates y la lógica backend del views.py. Además, en todas las plantillas creadas hay también código javascript asociado. Por otro lado, se ha creado otra aplicación para el manejo de usuario y la creación de estos. Cada vez que se crea un usuario se copia la estructura que hay en global (archivos de configuración, etc) y se guarda en una estructura de carpetas del estilo tenants/nombre_usuario/… donde los siguientes subdirectorios son parecidos a la estructura de global en el sentido de que cada usuario tiene sus modelos, su capeta de data, etc… , En este diseño compartimental se almacena la información que se va generando en la plataforma

# Tests:

Para ejecutar los tests, desde la raíz del proyecto:
python -m pytest tests

Para ejecutar un script en concreto:
python -m pytest tests/test_own_utils.py

Para ejecutar el test de una ejecución en concreto:
python -m pytest tests/test_own_utils.py::TestExecuteConcurrently