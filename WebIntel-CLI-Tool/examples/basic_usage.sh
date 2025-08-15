#!/bin/bash
# WebIntel Basic Usage Examples

echo "🚀 WebIntel Usage Examples"
echo "=========================="

# Set your API key
export OPENAI_API_KEY="your-api-key-here"

echo "1. Basic website audit"
python webintel.py --url https://stripe.com

echo "2. Audit with custom output directory"
mkdir -p competitive_reports
python webintel.py --url https://shopify.com --output competitive_reports/

echo "3. Verbose logging for debugging"
python webintel.py --url https://github.com --verbose

echo "4. Multiple competitor audits"
competitors=(
    "https://stripe.com"
    "https://square.com" 
    "https://paypal.com"
)

for url in "${competitors[@]}"; do
    echo "Auditing $url..."
    python webintel.py --url "$url" --output competitive_reports/
done

echo "✅ All audits completed!"
echo "📁 Check the competitive_reports/ directory for results"