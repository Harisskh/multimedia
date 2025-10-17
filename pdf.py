import nbformat
from nbconvert import HTMLExporter
import os
import re

def export_clean_pdf(notebook_file):
    """
    Export notebook ke PDF tanpa header, dengan error handling yang lebih baik
    """
    try:
        print("📖 Membaca notebook...")
        # Read notebook
        with open(notebook_file, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        print("🔄 Convert ke HTML...")
        # Create HTML exporter
        html_exporter = HTMLExporter()
        html_exporter.exclude_input_prompt = True
        html_exporter.exclude_output_prompt = True
        
        # Export to HTML
        (body, resources) = html_exporter.from_notebook_node(nb)
        
        print("🧹 Membersihkan HTML...")
        # Remove unwanted elements
        body = re.sub(r'<title>.*?</title>', '<title>Tugas</title>', body)
        body = re.sub(r'<h1[^>]*>.*?Notebook.*?</h1>', '', body, flags=re.IGNORECASE)
        body = re.sub(r'<p[^>]*>.*?September.*?</p>', '', body, flags=re.IGNORECASE)
        
        # Try to convert to PDF
        print("📄 Converting ke PDF...")
        try:
            from weasyprint import HTML, CSS
            
            output_file = notebook_file.replace('.ipynb', '_clean.pdf')
            
            # Custom CSS
            custom_css = CSS(string='''
                @page { margin: 0.8in; }
                body { font-family: Arial, sans-serif; }
                .container { max-width: 100% !important; }
            ''')
            
            HTML(string=body).write_pdf(output_file, stylesheets=[custom_css])
            print(f"✅ SUCCESS! PDF berhasil dibuat: {output_file}")
            return True
            
        except ImportError:
            print("❌ weasyprint belum terinstall!")
            print("💡 Install dengan: pip install weasyprint")
            print("⏳ Sementara menyimpan sebagai HTML...")
            
            # Save clean HTML instead
            html_file = notebook_file.replace('.ipynb', '_clean.html')
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(body)
            
            print(f"📄 HTML bersih tersimpan: {html_file}")
            print("🖨️  Cara manual: Buka HTML di browser → Print → Save as PDF")
            return False
            
        except Exception as pdf_error:
            print(f"❌ Error saat convert PDF: {pdf_error}")
            
            # Fallback to HTML
            html_file = notebook_file.replace('.ipynb', '_clean.html')
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(body)
            
            print(f"📄 Fallback: HTML tersimpan sebagai {html_file}")
            return False
            
    except Exception as e:
        print(f"❌ Error general: {e}")
        return False

# Main
if __name__ == "__main__":
    notebook_file = "122140040_handson_audio.ipynb"
    
    print("🚀 Starting Clean PDF Export...")
    print(f"📁 Target file: {notebook_file}")
    
    if not os.path.exists(notebook_file):
        print(f"❌ File {notebook_file} tidak ditemukan!")
        available_files = [f for f in os.listdir('.') if f.endswith('.ipynb')]
        if available_files:
            print("📂 Available notebooks:")
            for file in available_files:
                print(f"   - {file}")
    else:
        success = export_clean_pdf(notebook_file)
        
        if not success:
            print("\n🔧 SOLUSI:")
            print("1. Install weasyprint: pip install weasyprint")
            print("2. Atau manual: buka HTML yang sudah dibuat di browser")
            print("3. Print → Save as PDF")
    
    input("\n⏸️  Tekan Enter untuk keluar...")