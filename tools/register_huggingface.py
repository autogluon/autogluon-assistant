from autogluon.assistant.tools_registry.register_huggingface import HuggingFaceToolRegistrar


def main():
    """
    Main function to run the registration process.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Register Hugging Face as an ML tool")
    parser.add_argument(
        "--output-dir", default="hf_tutorials", help="Directory to save tutorial files (default: hf_tutorials)"
    )
    parser.add_argument(
        "--top-models", type=int, default=2, help="Number of top models to fetch per task (default: 2)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create registrar and run the process
    registrar = HuggingFaceToolRegistrar(output_dir=args.output_dir, top_models_per_task=args.top_models)

    registrar.run_registration_process()


if __name__ == "__main__":
    main()
