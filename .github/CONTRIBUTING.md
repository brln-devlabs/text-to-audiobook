# Contributing to text-to-audiobook

First off, thanks for taking the time to contribute! 🎉

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title**
* **Describe the exact steps which reproduce the problem**
* **Provide specific examples to demonstrate the steps**
* **Describe the behavior you observed after following the steps**
* **Explain which behavior you expected to see instead and why**
* **Include screenshots and animated GIFs if possible**
* **Include your environment details** (OS, Rust version, build features, GPU type, etc.)
* **Include relevant logs** from `logs/pipeline.log`

### Suggesting Enhancements

When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title**
* **Provide a step-by-step description of the suggested enhancement**
* **Provide specific examples to demonstrate the steps**
* **Describe the current behavior and what the expected behavior would be**
* **Explain why this enhancement would be useful**

### Pull Requests

* Fill in the required template
* Follow the Rust styleguide below
* Include appropriate test cases
* Update documentation as needed
* End all files with a newline

## Styleguides

### Rust Styleguide

* Format with `cargo fmt`
* Keep `cargo clippy --all-targets -- -D warnings` clean
* Use meaningful variable names
* Keep functions focused and single-purpose
* Add doc comments (`///`) to public items

### Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

### Documentation Styleguide

* Use Markdown for documentation
* Reference code blocks with triple backticks (```)
* Add code examples for new features
* Keep documentation up-to-date with code changes

## Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/text-to-audiobook.git`
3. Install Rust (https://rustup.rs/), CMake and a C/C++ compiler
4. Build: `cargo build`
5. Create a branch for your changes: `git checkout -b my-feature-branch`

## Testing

* Write tests for new features
* Ensure all existing tests pass: `cargo test`
* The tests use mock models; `cargo run -- --mock-models run` tries the whole pipeline without models
* Run `cargo fmt --check` and `cargo clippy --all-targets` before submitting a PR
