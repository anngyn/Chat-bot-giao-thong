# from __future__ import annotations

# import json

# import fitz  # PyMuPDF

# # Extract text from the PDF


# def extract_text_from_pdf(pdf_path):
#     text = ''
#     pdf_document = fitz.open(pdf_path)
#     for page_num in range(len(pdf_document)):
#         page = pdf_document.load_page(page_num)
#         text += page.get_text('text')  # Extract text from each page
#     return text


# def extract_sections_by_dieu(text, section_type='ATGT'):
#     lines = text.splitlines()
#     extracted_data = []
#     current_chuong = ''
#     current_muc = ''
#     current_dieu = ''
#     chuong_content = ''
#     muc_content = ''
#     dieu_content = ''

#     for line in lines:
#         line = line.strip()

#         # Detect "Chương"
#         if line.startswith('Chương'):
#             if current_dieu:  # Save previous Điều if available
#                 content = f'{chuong_content}\n{muc_content}\n{dieu_content}'
#                 extracted_data.append({
#                     'title': current_dieu,
#                     'content': content,
#                     'type': section_type,
#                 })
#             current_chuong = line  # Set current Chương
#             chuong_content = line  # Reset Chương content
#             current_muc = ''  # Reset Mục for a new Chương
#             muc_content = ''  # Reset Mục content
#             dieu_content = ''  # Reset Điều content

#         # Detect "Mục"
#         elif line.startswith('Mục'):
#             if current_dieu:
#                 content = f'{chuong_content}\n{muc_content}\n{dieu_content}'
#                 # Save previous Điều if available
#                 extracted_data.append({
#                     'title': current_dieu,
#                     'content': content,
#                     'type': section_type,
#                 })
#             current_muc = line  # Set current Mục
#             muc_content = line  # Reset Mục content
#             dieu_content = ''  # Reset Điều content

#         # Detect "Điều"
#         elif line.startswith('Điều'):
#             content = f'{chuong_content}\n{muc_content}\n{dieu_content}'
#             if current_dieu:
#                 extracted_data.append({
#                     'title': current_dieu,
#                     'content': content,
#                     'type': section_type,
#                 })
#             # Combine Chương, Mục, and Điều as title
#             current_dieu = f'{current_chuong} {current_muc} {line}'
#             dieu_content = line  # Initialize content for Điều

#         # Otherwise, accumulate content
#         else:
#             if current_dieu:  # Add content for Điều
#                 dieu_content += '\n' + line
#             elif current_muc:  # Add content for Mục
#                 muc_content += '\n' + line
#             elif current_chuong:  # Add content for Chương
#                 chuong_content += '\n' + line

#     # Save the last Điều after processing all lines
#     if current_dieu:
#         extracted_data.append({
#             'title': current_dieu,
#             'content': f'{chuong_content}\n{muc_content}\n{dieu_content}',
#             'type': section_type,
#         })

#     return extracted_data

# # Save extracted data to JSON file


# def save_to_json(data, output_json):
#     with open(output_json, mode='a', encoding='utf-8') as file:
#         json.dump(data, file, ensure_ascii=False, indent=4)


# # Path to the PDF file
# # Replace with the actual PDF path
# pdf_path = r'.\AI002\data\36_2024_QH15_444251.pdf'

# # Path to the output JSON file
# output_json = r'AI002\data\output.json'

# # Extract text from the PDF
# text = extract_text_from_pdf(pdf_path)
# section_type = 'DB'  # You can set it manually to "ATGT" or "DB"
# data = extract_sections_by_dieu(text, section_type)

# # Save the extracted data to the JSON file
# save_to_json(data, output_json)

# print(f'Data successfully saved to {output_json}')


# crawl_data.py
from __future__ import annotations

import json
import os
import fitz  # PyMuPDF
import re # Cần cho regex pattern trong hàm mới

# Import SentenceSplitter từ llama_index
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document # Cần để tạo Document cho SentenceSplitter

# Extract text from the PDF
def extract_text_from_pdf(pdf_path):
    text = ''
    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text('text')  # Extract text from each page
    pdf_document.close()
    return text

# Cập nhật hàm extract_sections_by_dieu để bao gồm SentenceSplitter
def extract_sections_by_dieu(text, section_type='ATGT', max_chunk_size=1000, chunk_overlap=200):
    lines = text.splitlines()
    extracted_data = []
    
    current_chuong = ''
    current_muc = ''
    current_dieu = ''
    dieu_content_buffer = [] # Buffer để chứa nội dung của Điều hiện tại

    # Khởi tạo SentenceSplitter
    sentence_splitter = SentenceSplitter(chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)

    def process_and_add_chunk(title, content_buffer):
        """Xử lý buffer nội dung và thêm vào extracted_data sau khi chia nhỏ."""
        if not content_buffer:
            return

        full_content = "\n".join(content_buffer).strip()
        if not full_content: # Tránh xử lý nội dung rỗng
            return

        # Tạo Document từ nội dung đã gom được
        doc_to_split = Document(text=full_content)
        
        # Chia nhỏ Document thành nhiều nodes (chunks) nhỏ hơn
        nodes = sentence_splitter.get_nodes_from_documents([doc_to_split])
        
        for node in nodes:
            # Gắn metadata (tiêu đề và loại) cho từng chunk
            # Tiêu đề của chunk con sẽ là tiêu đề của Điều/Mục/Chương cha
            extracted_data.append({
                'title': title, 
                'content': node.text.strip(),
                'type': section_type,
            })

    # --- Logic chính của hàm ---
    for line in lines:
        line = line.strip()
        if not line: # Bỏ qua dòng trống
            continue

        # Detect "Chương"
        if re.match(r'Chương\s+[IVX\d]+\b', line, re.IGNORECASE): # Regex linh hoạt hơn cho Chương
            # Lưu Điều/Mục/Chương trước đó
            if current_dieu:
                process_and_add_chunk(current_dieu, dieu_content_buffer)
            elif current_muc: # Nếu không có Điều, có thể là nội dung thuộc Mục
                process_and_add_chunk(f'{current_chuong} {current_muc}', dieu_content_buffer)
            elif current_chuong: # Nếu không có Điều/Mục, có thể là nội dung thuộc Chương
                process_and_add_chunk(current_chuong, dieu_content_buffer)
            
            # Reset cho Chương mới
            current_chuong = line
            current_muc = ''
            current_dieu = ''
            dieu_content_buffer = [] # Reset buffer
            dieu_content_buffer.append(line) # Thêm dòng Chương vào buffer
        
        # Detect "Mục"
        elif re.match(r'Mục\s+[IVX\d]+\b', line, re.IGNORECASE): # Regex linh hoạt hơn cho Mục
            # Lưu Điều/Mục/Chương trước đó
            if current_dieu:
                process_and_add_chunk(current_dieu, dieu_content_buffer)
            elif current_muc: # Nếu không có Điều, có thể là nội dung thuộc Mục
                process_and_add_chunk(f'{current_chuong} {current_muc}', dieu_content_buffer)

            # Reset cho Mục mới
            current_muc = line
            current_dieu = ''
            dieu_content_buffer = [] # Reset buffer
            dieu_content_buffer.append(line) # Thêm dòng Mục vào buffer
        
        # Detect "Điều"
        elif re.match(r'Điều\s+\d+\.\s*', line, re.IGNORECASE): # Regex cho Điều
            # Lưu Điều trước đó
            if current_dieu:
                process_and_add_chunk(current_dieu, dieu_content_buffer)
            
            # Tạo tiêu đề đầy đủ cho Điều mới
            full_title = current_chuong
            if current_muc:
                full_title += f" {current_muc}"
            full_title += f" {line}"
            current_dieu = full_title.strip()
            
            dieu_content_buffer = [] # Reset buffer
            dieu_content_buffer.append(line) # Thêm dòng Điều vào buffer
        
        # Otherwise, accumulate content
        else:
            # Thêm dòng vào buffer của Điều/Mục/Chương hiện tại
            dieu_content_buffer.append(line)

    # Xử lý và lưu chunk cuối cùng sau khi đã duyệt hết tất cả các dòng
    if current_dieu:
        process_and_add_chunk(current_dieu, dieu_content_buffer)
    elif current_muc:
        process_and_add_chunk(f'{current_chuong} {current_muc}', dieu_content_buffer)
    elif current_chuong:
        process_and_add_chunk(current_chuong, dieu_content_buffer)
    # Xử lý nếu có nội dung mà không có Chương/Mục/Điều nào (có thể là phần mở đầu/kết thúc)
    elif dieu_content_buffer:
        process_and_add_chunk("Tổng quan/Khác", dieu_content_buffer) # Tạo tiêu đề chung

    return extracted_data

# Save extracted data to JSON file
def save_to_json(data, output_json):
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f'Data successfully saved to {output_json}')

# --- Logic chính để crawl và xử lý các file PDF ---
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(current_dir, '..', '..')

    pdf_paths = [
        os.path.join(root_dir, 'data', '36-2024-qh15.pdf'),
        os.path.join(root_dir, 'data', '36-2024-qh15_tiep.pdf'),
        #os.path.join(root_dir, 'data', '100-2019-nd-cp.pdf'), # Thêm file Nghị định mới
        
    ]

    output_json_file = os.path.join(root_dir, 'data', 'output.json')

    all_extracted_data = []
    
    # Xóa file output.json cũ nếu tồn tại để tránh lỗi JSON khi ghi đè
    if os.path.exists(output_json_file):
        os.remove(output_json_file)
        print(f"Đã xóa file JSON cũ: {output_json_file}")

    for pdf_path in pdf_paths:
        print(f"Đang trích xuất dữ liệu từ: {pdf_path}")
        text = extract_text_from_pdf(pdf_path)
        
        # Đặt một loại chung cho tất cả hoặc phân biệt tùy theo nhu cầu phân loại sau này
        section_type = 'LUAT_GTDB' 
        
        # Sử dụng hàm extract_sections_by_dieu đã cải tiến
        data = extract_sections_by_dieu(text, section_type=section_type, max_chunk_size=1000, chunk_overlap=200)
        all_extracted_data.extend(data)
    
    save_to_json(all_extracted_data, output_json_file)