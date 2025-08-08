# AutoGluon-Assistant Documentation

This directory contains the documentation for AutoGluon-Assistant.

## Building the Documentation

To build the documentation locally:

1. Install the documentation requirements:
   ```bash
   pip install -r requirements_doc.txt
   ```

2. Install AutoGluon-Assistant in development mode:
   ```bash
   pip install -e ..
   ```

3. Build the documentation:
   ```bash
   sphinx-build -b html . _build/html
   ```

4. Open the documentation in your browser:
   ```bash
   open _build/html/index.html
   ```

## Documentation Structure

- `index.md` - Main landing page
- `api/` - API reference documentation
- `tutorials/` - Step-by-step tutorials
- `whats_new/` - Release notes and changelog
- `_static/` - Static assets (images, CSS, JS)
- `_templates/` - Sphinx templates
- `conf.py` - Sphinx configuration

## Contributing

When adding new documentation:

1. Follow the existing structure and style
2. Use MyST Markdown format for content
3. Add new pages to the appropriate toctree
4. Test the build locally before submitting
5. Update the API documentation if adding new modules
