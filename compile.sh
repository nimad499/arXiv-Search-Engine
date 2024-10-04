cd lib/pdf_to_text
go get "github.com/heussd/pdftotext-go"
go build -buildmode=c-shared -o pdf_to_text.so main.go