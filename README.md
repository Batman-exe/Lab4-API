# Lab4-API

Para ejecutar se debe abrir la consola de anaconda ir a la carpeta raíz de este proyecto y ejecutar el comando:  

```uvicorn main:app --reload``` 

Para obtener la predicción de uno o más elementos se debe entrar a http://127.0.0.1:8000/docs   
  
En la seccion post /predictions > try it out se debe poner un body en formato JSON con la cantidad de elementos que deseemos de esta manera:  
```json
[   
  {  
    "serial_no": 0,  
    "gre_score": 0,  
    "toefl_score": 0,  
    "university_rating": 0,  
    "sop": 0,  
    "lor": 0,  
    "cgpa": 0,  
    "research": 0  
  },  
  ... ,  
  {  
    "serial_no": 479,  
    "gre_score": 327,  
    "toefl_score": 113,  
    "university_rating": 4,  
    "sop": 4.0,  
    "lor": 2.77,  
    "cgpa": 8.88,  
    "research": 1  
  }  
]  
```

Para reentrenar el pipeline se debe hacer en la sección de post /train > try it out, el formato debe ser igual al anterior, pero debe incluir la tupla ```"admission_points": value```,  en cada objeto del body.
