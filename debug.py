from utils.manager import PortfolioManager

def main():
    manager = PortfolioManager(100, 0.0, 4, 2, 3, 3)
    manager.print_portfolio(1)
    manager.sell(30, 1)    
    manager.print_portfolio(1)

if __name__ == '__main__':
    main()