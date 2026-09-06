# Cómo subir este código a GitHub

## Opción sencilla: desde la web de GitHub

1. Entra en GitHub e inicia sesión.
2. Pulsa **New repository**.
3. Nombre sugerido:
   `FCE-HFCE-analysis`
4. Descripción sugerida:
   `Reproducibility code and analytical extensions for fuzzy combination entropy and HFCE outlier detection.`
5. Inicialmente puedes dejarlo como **Private** mientras preparáis el artículo.
6. No marques "Add a README" si vas a subir directamente este paquete, porque ya contiene uno.
7. Crea el repositorio.
8. En la página del repositorio, usa **Add file > Upload files**.
9. Sube el contenido de la carpeta `HFCE_FCE_PatternRecognition_code` manteniendo las carpetas `src`, `results` y `docs`.
10. Haz el commit.

No subas los datasets de los autores salvo que su licencia permita redistribuirlos. Es mejor enlazar su repositorio y explicar en el README dónde colocarlos.

## Opción recomendada: Git desde tu ordenador

Descomprime el paquete y abre una terminal dentro de la carpeta.

```bash
git init
git add .
git commit -m "Initial reproducibility package"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/FCE-HFCE-analysis.git
git push -u origin main
```

GitHub puede pedir autenticación mediante navegador o token según tu configuración.

## Antes de hacerlo público

Conviene comprobar:

- que no hay datasets que no tengáis permiso para redistribuir;
- que no hay rutas personales o locales;
- que README explica cómo obtener los datos;
- que `requirements.txt` está incluido;
- que los resultados esenciales pueden reproducirse;
- que elegís una licencia para vuestro código;
- que el nombre final del artículo y autores están actualizados.

## Para el manuscrito

Una vez público, podéis añadir:

```latex
\section*{Data availability}

No new datasets were generated in this study. The 20 benchmark datasets
analyzed in the experimental assessment are the publicly released datasets
accompanying the HFCE implementation cited in \cite{Su2026}. The code used
to reproduce the HFCE benchmark results and to evaluate the proposed
analytical extensions, together with the exact experimental configurations,
is publicly available at \url{https://github.com/TU_USUARIO/FCE-HFCE-analysis}.
```

Si preferís no hacer público el código antes de la aceptación, podéis mantener
el repositorio privado durante la revisión y sustituir temporalmente la última
frase por:

```text
The code and exact configurations used in the reproduction and extension
experiments will be made publicly available upon publication.
```
