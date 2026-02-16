
您说得对，对于 Tushare 确实是**调用标准 API**，而不是爬网页。

我的代码命名 `Crawler` 是为了保持和项目中原有 `EastmoneyCrawler`（它是真爬虫）的命名一致，实际上我的实现完全是**调用 Tushare SDK 接口**：
1. `pro.fund_share()` 获取份额
2. `pro.moneyflow()` 获取资金流

所以准确来说是 **“Tushare 数据获取器”**。

只要配置了 Token，就可以直接通过 API 获取，不需要任何爬虫逻辑（解析 HTML 等）。

Tushare Token: `3231795730303233c0bd4007df945283857b2803b98c397554580bf1`
