# WebIntel - AI-Powered Competitive Website Audit Tool 🚀

WebIntel is a powerful CLI tool that performs automated competitive analysis of websites using AI-powered content summarization. Generate detailed markdown reports analyzing competitors' product features, pricing strategies, design patterns, and value propositions.

## ✨ Features

- **🎯 Smart Content Extraction**: Extracts relevant website content while skipping ads, navbars, footers, and clutter
- **🤖 Hugging Face AI Summarization**: Uses BART (`facebook/bart-large-cnn`) for clean, high-level summaries
- **📊 Structured Markdown Reports**: Covers pricing strategy, product offerings, brand tone, and UX patterns
- **🛠️ Zero API Keys Needed**: Fully local — no OpenAI API required
- **🧱 Modular Architecture**: Cleanly separated scraping, parsing, and summarization logic
- **📁 CLI-First**: Lightweight and fast command-line experience

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/WebIntel-CLI-Tool.git
cd WebIntel-CLI-Tool

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### 2. Run Your First Audit

```bash
# Basic usage
python webintel.py --url https://stripe.com

# Specify output directory
python webintel.py --url shopify.com --output reports/

# Verbose logging
python webintel.py --url https://openai.com --verbose
```

## 📋 Usage Examples

### Basic Website Audit
```bash
python webintel.py --url https://stripe.com
```

### Batch Processing Multiple Sites
```bash
# Create a batch script
python webintel.py --url https://stripe.com --output competitive_reports/
python webintel.py --url https://square.com --output competitive_reports/
python webintel.py --url https://paypal.com --output competitive_reports/
```

### Advanced Options
```bash
# Save reports to a directory
python webintel.py --url https://openai.com --output reports/

# Use a custom Hugging Face model
python webintel.py --url https://huggingface.co --model facebook/bart-large-cnn

```

## 📊 Sample Output

WebIntel generates comprehensive markdown reports like `stripe_audit.md`:

```markdown
# WebIntel Competitive Audit: Stripe

**Generated:** 2025-08-07 10:30:00 UTC
**Target URL:** https://stripe.com
**Tool:** WebIntel v1.0

## Company Overview
- Leading payment processing platform for online businesses
- Targets developers and growing companies with API-first approach

## Product/Service Analysis
- Core payment processing APIs and SDKs
- Advanced features: Connect, Billing, Terminal, Radar
- Developer-centric tools and documentation

## Pricing Strategy
- Transparent per-transaction pricing (2.9% + 30¢)
- No monthly fees or setup costs
- Volume discounts for enterprise customers
...
```

## 🔧 Configuration

### Environment Variables
```bash
WEBINTEL_TIMEOUT=30                 # Optional: Request timeout in seconds
WEBINTEL_MAX_RETRIES=3              # Optional: Maximum retry attempts
```

### CLI Arguments
```
--url        Target website URL (required)
--output     Output directory for report files (default: current directory)
--model      Hugging Face summarization model (default: facebook/bart-large-cnn)
--help       Show usage
```

## 🧠 Supported Models
This project uses Hugging Face’s transformers library. Default model:
facebook/bart-large-cnn

You can try other summarization models:

- **sshleifer/distilbart-cnn-12-6**
- **google/pegasus-xsum**
- **mistralai/Mistral-7B-Instruct-v0.2 (via transformers + accelerate)**

## 🏗️ Architecture

WebIntel is built with a modular architecture:

- **WebIntelScraper**: Handles website fetching with retry logic and error handling
- **ContentProcessor**: Cleans HTML, extracts main content, converts to markdown
- **AIAnalyzer**: Interfaces with Hugging Face pipelines for intelligent content analysis
- **WebIntel**: Main orchestrator that coordinates the audit process

## 🧪 Testing

```bash
# Run basic tests
python -m pytest tests/ -v

pytest tests/ -v
# Test specific components
python -m pytest tests/test_content_processor.py -v
```

## 📈 Roadmap

- [ ] Multi-page crawling support
- [ ] Screenshot capture and visual analysis
- [ ] Competitive comparison reports (side-by-side)
- [ ] Integration with popular business intelligence tools
- [ ] Custom analysis templates
- [ ] Export to PDF, Excel, and other formats

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

WebIntel is designed for competitive research and analysis purposes. Please ensure you comply with websites' robots.txt files and terms of service when using this tool. Always respect rate limits and use responsibly.

## 🆘 Support

- **Issues**: Report bugs and request features on GitHub Issues
- **Email**: maanaskarthikeyan@gmail.com

---

**Made with ❤️ for competitive intelligence professionals**