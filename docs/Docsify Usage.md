# Docsify Usage

This is a simple note on how to use Docsify, a documentation generator that creates a website from Markdown files.

## Installation
To use Docsify, you need to have Node.js installed on your machine. 
You can then install Docsify globally using npm:

```bash
npm install -g docsify-cli
```

## Creating a New Docsify Project
To create a new Docsify project, navigate to the directory where you want to create your documentation and run:

```bash
docsify init ./docs
or
docsify init .
```

This will create a new directory called `docs` (or use the current directory if you used `.`) with the necessary files.

## Running the Docsify Server
To preview your documentation, navigate to the `docs` directory and run:

```bash
docsify serve .
```

This will start a local server, and you can view your documentation by opening `http://localhost:3000` in your web browser.
