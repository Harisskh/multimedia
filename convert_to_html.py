import nbformat
from nbconvert import HTMLExporter
import os
import re

def export_clean_html(notebook_file):
    """
    Export notebook ke file HTML yang bersih.
    """
    try:
        print("📖 Membaca notebook...")
        # Baca notebook
        with open(notebook_file, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        print("🔄 Convert ke HTML...")
        # Buat HTML exporter
        html_exporter = HTMLExporter()
        html_exporter.exclude_input_prompt = True
        html_exporter.exclude_output_prompt = True
        
        # Export ke HTML
        (body, resources) = html_exporter.from_notebook_node(nb)
        
        print("🧹 Membersihkan HTML...")
        # Bersihkan elemen yang tidak diinginkan
        body = re.sub(r'<title>.*?</title>', '<title>Tugas Analisis Multimedia</title>', body)
        body = re.sub(r'<h1[^>]*>.*?Notebook.*?</h1>', '', body, flags=re.IGNORECASE)
        
        # Simpan HTML
        output_file = notebook_file.replace('.ipynb', '_clean.html')
        
        print(f"📄 Menyimpan HTML ke: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(body)
            
        print(f"✅ BERHASIL! File HTML: {output_file}")
        return output_file
        
    except FileNotFoundError:
        print(f"❌ Error: File '{notebook_file}' tidak ditemukan.")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Ganti dengan nama file notebook Anda
    notebook_file = "122140040_worksheet4/122140040_worksheet4.ipynb"
    
    print("🚀 Memulai konversi IPYNB ke HTML...\n")
    
    if not os.path.exists(notebook_file):
        print(f"❌ File '{notebook_file}' tidak ditemukan!")
        print("\n📂 File .ipynb yang tersedia:")
        for file in os.listdir('.'):
            if file.endswith('.ipynb'):
                print(f"   • {file}")
    else:
        result = export_clean_html(notebook_file)
        
        if result:
            print(f"\n✨ Selesai! Buka file: {result}")
        else:
            print("\n❌ Konversi gagal.")
    
    input("\n⏸️  Tekan Enter untuk keluar...")