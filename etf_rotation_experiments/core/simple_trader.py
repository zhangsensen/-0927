import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class SimpleTrader:
    """
    Simple Paper Trading System
    - Manages Account State (Cash, Holdings)
    - Records Transaction Ledger
    - Tracks Daily NAV
    """
    def __init__(self, data_dir: Path, initial_capital: float = 100000.0):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.account_file = self.data_dir / "account.json"
        self.ledger_file = self.data_dir / "ledger.csv"
        self.nav_file = self.data_dir / "nav.csv"
        
        self.initial_capital = initial_capital
        self.load_state()

    def load_state(self):
        """Load account state or initialize if new."""
        if self.account_file.exists():
            with open(self.account_file, 'r') as f:
                self.account = json.load(f)
        else:
            self.account = {
                "cash": self.initial_capital,
                "holdings": {},  # Ticker -> Quantity
                "initial_capital": self.initial_capital,
                "created_at": datetime.now().isoformat()
            }
            self.save_state()
            
        # Initialize Ledger if needed
        if not self.ledger_file.exists():
            pd.DataFrame(columns=[
                "timestamp", "date", "ticker", "action", 
                "price", "quantity", "amount", "fee", "cash_balance"
            ]).to_csv(self.ledger_file, index=False)

        # Initialize NAV if needed
        if not self.nav_file.exists():
            pd.DataFrame(columns=[
                "date", "total_value", "cash", "holdings_value", "return_pct"
            ]).to_csv(self.nav_file, index=False)

    def save_state(self):
        """Persist account state to JSON."""
        with open(self.account_file, 'w') as f:
            json.dump(self.account, f, indent=2)

    def get_holdings(self) -> Dict[str, int]:
        return self.account["holdings"]

    def get_cash(self) -> float:
        return self.account["cash"]

    def calculate_nav(self, current_prices: Dict[str, float]) -> float:
        """Calculate Net Asset Value based on current market prices."""
        holdings_val = 0.0
        for ticker, qty in self.account["holdings"].items():
            price = current_prices.get(ticker)
            if price:
                holdings_val += qty * price
            else:
                # Fallback or warning? For now, assume 0 if price missing (dangerous but loud)
                print(f"⚠️ Warning: No price found for {ticker}, using 0 value.")
        
        return self.account["cash"] + holdings_val

    def generate_rebalance_orders(self, target_weights: Dict[str, float], current_prices: Dict[str, float]) -> List[Dict]:
        """
        Generate a list of orders to achieve target weights.
        Sell orders are generated first to free up cash.
        """
        nav = self.calculate_nav(current_prices)
        orders = []
        
        # 1. Calculate Target Value per Ticker
        target_values = {t: nav * w for t, w in target_weights.items()}
        
        # 2. Identify Sells
        current_holdings = self.account["holdings"]
        for ticker, qty in current_holdings.items():
            price = current_prices.get(ticker)
            if not price: continue
            
            current_val = qty * price
            target_val = target_values.get(ticker, 0.0)
            
            if current_val > target_val:
                # Sell difference
                diff = current_val - target_val
                # Sell at least 100 shares (1 lot) logic? Or exact?
                # Let's do exact for now, or floor to 100 if A-shares.
                # A-shares: Buy in 100s, Sell can be odd (but usually sell all odd lots).
                # Simplified: Floor to 100 for Buy, Exact for Sell (or floor to 100).
                # Let's stick to 100 lots for simplicity and realism.
                
                sell_qty = int(diff / price / 100) * 100
                if sell_qty > 0:
                    orders.append({
                        "action": "SELL",
                        "ticker": ticker,
                        "quantity": sell_qty,
                        "price": price,
                        "reason": "Rebalance"
                    })

        # 3. Identify Buys (After Sells)
        # Estimate cash after sells
        estimated_cash = self.account["cash"]
        for o in orders:
            if o["action"] == "SELL":
                estimated_cash += o["quantity"] * o["price"] # Ignoring fees for estimation
        
        for ticker, target_val in target_values.items():
            price = current_prices.get(ticker)
            if not price: continue
            
            current_qty = current_holdings.get(ticker, 0)
            current_val = current_qty * price
            
            if current_val < target_val:
                diff = target_val - current_val
                buy_qty = int(diff / price / 100) * 100
                
                if buy_qty > 0:
                    cost = buy_qty * price
                    if estimated_cash >= cost:
                        orders.append({
                            "action": "BUY",
                            "ticker": ticker,
                            "quantity": buy_qty,
                            "price": price,
                            "reason": "Rebalance"
                        })
                        estimated_cash -= cost
        
        return orders

    def execute_order(self, order: Dict, fee_rate: float = 0.0003, min_fee: float = 5.0):
        """
        Execute a single order and update state.
        fee_rate: 0.03% commission (example)
        """
        action = order["action"]
        ticker = order["ticker"]
        qty = order["quantity"]
        price = order["price"]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        amount = qty * price
        
        # Calculate Fee
        # A-shares: Commission (min 5) + Stamp Duty (0.05% Sell only) + Transfer Fee
        commission = max(amount * fee_rate, min_fee)
        stamp_duty = amount * 0.0005 if action == "SELL" else 0.0
        total_fee = commission + stamp_duty
        
        if action == "BUY":
            cost = amount + total_fee
            if self.account["cash"] < cost:
                print(f"❌ Failed to BUY {ticker}: Insufficient Cash ({self.account['cash']:.2f} < {cost:.2f})")
                return False
            
            self.account["cash"] -= cost
            self.account["holdings"][ticker] = self.account["holdings"].get(ticker, 0) + qty
            
        elif action == "SELL":
            current_qty = self.account["holdings"].get(ticker, 0)
            if current_qty < qty:
                print(f"❌ Failed to SELL {ticker}: Insufficient Holdings ({current_qty} < {qty})")
                return False
                
            proceeds = amount - total_fee
            self.account["cash"] += proceeds
            self.account["holdings"][ticker] -= qty
            if self.account["holdings"][ticker] == 0:
                del self.account["holdings"][ticker]
        
        self.save_state()
        
        # Log
        record = {
            "timestamp": timestamp,
            "date": date_str,
            "ticker": ticker,
            "action": action,
            "price": price,
            "quantity": qty,
            "amount": amount,
            "fee": total_fee,
            "cash_balance": self.account["cash"]
        }
        
        # Append to CSV
        pd.DataFrame([record]).to_csv(self.ledger_file, mode='a', header=False, index=False)
        print(f"✅ Executed {action} {ticker}: {qty} @ {price:.3f} (Fee: {total_fee:.2f})")
        return True

    def log_daily_nav(self, current_prices: Dict[str, float]):
        """Record daily NAV snapshot."""
        nav = self.calculate_nav(current_prices)
        holdings_val = nav - self.account["cash"]
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Calculate return if possible
        # Simple return based on initial capital for now, or daily change if we loaded history
        ret_pct = (nav - self.initial_capital) / self.initial_capital * 100
        
        record = {
            "date": date_str,
            "total_value": nav,
            "cash": self.account["cash"],
            "holdings_value": holdings_val,
            "return_pct": ret_pct
        }
        
        # Check if date already exists to avoid duplicates
        df = pd.read_csv(self.nav_file)
        if date_str in df['date'].values:
            # Update existing row
            df.loc[df['date'] == date_str, ["total_value", "cash", "holdings_value", "return_pct"]] = [nav, self.account["cash"], holdings_val, ret_pct]
            df.to_csv(self.nav_file, index=False)
        else:
            pd.DataFrame([record]).to_csv(self.nav_file, mode='a', header=False, index=False)
        
        print(f"📊 Daily NAV Logged: {nav:.2f} (Ret: {ret_pct:.2f}%)")
