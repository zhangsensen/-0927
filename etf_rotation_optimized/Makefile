.PHONY: install run clean clean-results help

install:
	@echo "[install] 使用pep517安装本地包"
	@pip install -e .

run:
	@echo "[run] 运行ETF轮动流水线"
	@python run_final_production.py --config configs/default.yaml

clean:
	@echo "[clean] 删除缓存与编译产物"
	@rm -rf .cache __pycache__ */__pycache__

clean-results:
	@echo "[clean-results] 清理results目录下的产物"
	@find results -mindepth 1 -maxdepth 1 -print -exec rm -rf {} +

help:
	@echo "可用命令:"
	@echo "  make install        安装/更新依赖"
	@echo "  make run            执行完整流水线"
	@echo "  make clean          删除缓存与编译文件"
	@echo "  make clean-results  清空 results/ 输出"
