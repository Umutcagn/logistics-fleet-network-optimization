import importlib
import src.ui.dashboard as dashboard_module

importlib.reload(dashboard_module)

if __name__ == "__main__":
    dashboard_module.main()
