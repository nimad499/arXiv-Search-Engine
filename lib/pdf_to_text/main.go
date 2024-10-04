package main

import (
	"C"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/heussd/pdftotext-go"
)
import (
	"fmt"
)

func pages_to_txt(pdf_pages []pdftotext.PdfPage, txt_file_path string) error {
	file, err := os.Create(txt_file_path)
	if err != nil {
		return err
	}
	defer file.Close()

	merged_txt := ""
	for _, p := range pdf_pages {
		merged_txt += p.Content
	}

	_, err = file.WriteString(merged_txt)
	if err != nil {
		return err
	}

	return nil
}

func list_pdfs(directory string) ([]string, error) {
	var pdf_files []string
	err := filepath.WalkDir(directory, func(path string, info fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && filepath.Ext(info.Name()) == ".pdf" {
			pdf_files = append(pdf_files, path)
		}
		return nil
	})

	return pdf_files, err
}

func file_name_without_extension(path string) string {
	baseName := filepath.Base(path)

	return strings.TrimSuffix(baseName, filepath.Ext(baseName))
}

func extract_text_from_pdfs(pdf_dir string, txt_dir string) error {
	pdfs, err := list_pdfs(pdf_dir)
	if err != nil {
		return err
	}

	for _, p := range pdfs {
		pdf_file, _ := os.ReadFile(p)

		pages, _ := pdftotext.Extract(pdf_file)

		txt_file_path := txt_dir + "/" + file_name_without_extension(p) + ".txt"
		err := pages_to_txt(pages, txt_file_path)
		if err != nil {
			return err
		}
	}

	return nil
}

//export extract_text_from_pdfs_c
func extract_text_from_pdfs_c(pdf_dir *C.char, txt_dir *C.char) int {
	pdf_dir_str := C.GoString(pdf_dir)
	txt_dir_str := C.GoString(txt_dir)

	err := extract_text_from_pdfs(pdf_dir_str, txt_dir_str)

	if err != nil {
		fmt.Println(err)
		return 1
	}
	return 0
}

func main() {}
