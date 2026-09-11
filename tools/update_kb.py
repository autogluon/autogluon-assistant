#!/usr/bin/env python3
"""Non-interactive knowledge-base refresh for the AutoGluon tools.

Given a checkout of the autogluon docs tutorials (``docs/tutorials``), this:
  1. converts each tool's ``*.ipynb`` tutorials to Markdown,
  2. clears the tool's existing ``tutorials/`` and ``condensed_tutorials/``,
  3. re-condenses every tutorial with the mlzero LLM (reusing the exact,
     tested ``ToolsRegistry.add_tool_tutorials`` write format) -- parallelised
     at the file level by pointing it at single-file source dirs, and
  4. optionally clears + rebuilds the FAISS indices so they are shipped fresh.

Example:
    python tools/update_kb.py \
        --docs-tutorials /tmp/ag161/docs/tutorials \
        --provider-config src/autogluon/assistant/configs/bedrock.yaml \
        --rebuild-index
"""
import argparse
import logging
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_ipynb_to_md import batch_convert_notebooks  # noqa: E402

from autogluon.assistant.tools_registry import ToolsRegistry  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("update_kb")

# Map registry tool name -> subdirectory under docs/tutorials
TOOL_AREAS = {
    "autogluon.tabular": "tabular",
    "autogluon.timeseries": "timeseries",
    "autogluon.multimodal": "multimodal",
}


def _condense_one(registry, tool_name, md_root: Path, md_file: Path, llm_config, max_length):
    """Condense a single tutorial file by handing add_tool_tutorials a
    single-file source dir that preserves the file's relative path."""
    rel = md_file.relative_to(md_root)
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(md_file, dest)
        registry.add_tool_tutorials(
            tool_name=tool_name,
            tutorials_source=tmp,
            condense=True,
            llm_config=llm_config,
            max_length=max_length,
        )
    return str(rel)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs-tutorials", required=True, help="Path to autogluon/docs/tutorials")
    ap.add_argument(
        "--provider-config",
        default="src/autogluon/assistant/configs/bedrock.yaml",
        help="YAML config exposing an `llm:` section used for condensation",
    )
    ap.add_argument("--tools", nargs="+", default=list(TOOL_AREAS), help="Which tools to refresh")
    ap.add_argument("--max-length", type=int, default=32768, help="Max length for condensed tutorials")
    ap.add_argument("--workers", type=int, default=8, help="Parallel condensation workers")
    ap.add_argument("--rebuild-index", action="store_true", help="Clear + rebuild FAISS indices at the end")
    args = ap.parse_args()

    docs = Path(args.docs_tutorials)
    if not docs.exists():
        sys.exit(f"docs-tutorials path not found: {docs}")

    llm_config = OmegaConf.load(args.provider_config).llm
    registry = ToolsRegistry()
    registry.list_tools()  # warm the cache before threads read it

    tmp_md_root = Path(tempfile.mkdtemp(prefix="kb_md_"))
    jobs = []  # (tool_name, md_root, md_file)
    for tool in args.tools:
        area = TOOL_AREAS.get(tool)
        if area is None:
            logger.warning("No docs area mapping for %s -- skipping", tool)
            continue
        src_area = docs / area
        if not src_area.exists():
            logger.warning("Missing docs area %s -- skipping %s", src_area, tool)
            continue

        md_area = tmp_md_root / area
        logger.info("[%s] converting notebooks %s -> %s", tool, src_area, md_area)
        batch_convert_notebooks(str(src_area), str(md_area))

        # Clear existing tutorials so stale/removed files do not linger
        tool_path = registry.get_tool_path(tool)
        for sub in ("tutorials", "condensed_tutorials"):
            d = tool_path / sub
            if d.exists():
                shutil.rmtree(d)

        for md_file in sorted(md_area.rglob("*.md")):
            if md_file.name == "index.md":
                continue
            jobs.append((tool, md_area, md_file))

    logger.info("Condensing %d tutorials across %d tools with %d workers", len(jobs), len(args.tools), args.workers)
    done, failed = 0, []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_condense_one, registry, t, r, f, llm_config, args.max_length): (t, f) for t, r, f in jobs}
        for fut in as_completed(futs):
            tool, md_file = futs[fut]
            try:
                fut.result()
                done += 1
                logger.info("[%d/%d] condensed %s :: %s", done, len(jobs), tool, md_file.name)
            except Exception as e:  # noqa: BLE001
                failed.append((tool, md_file, repr(e)))
                logger.error("FAILED %s :: %s -- %s", tool, md_file.name, e)

    logger.info("Condensation done in %.1fs: %d ok, %d failed", time.time() - t0, done, len(failed))
    for tool, f, err in failed:
        logger.error("  - %s %s: %s", tool, f, err)

    shutil.rmtree(tmp_md_root, ignore_errors=True)

    if args.rebuild_index:
        from autogluon.assistant.tools_registry.indexing import TutorialIndexer

        indices_dir = Path(registry.registry_path) / "indices"
        if indices_dir.exists():
            logger.info("Clearing stale indices at %s", indices_dir)
            shutil.rmtree(indices_dir)
        logger.info("Rebuilding FAISS indices ...")
        indexer = TutorialIndexer()
        indexer.build_indices()
        indexer.save_indices()
        indexer.cleanup()
        logger.info("Indices rebuilt.")

    logger.info("KB refresh complete.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
