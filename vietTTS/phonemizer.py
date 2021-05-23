# paper: https://www.aclweb.org/anthology/W16-5207.pdf
# title: A non-expert Kaldi recipe for Vietnamese Speech Recognition System

import unicodedata
consonants = [
    'ngh',
    'ch', 'gh', 'gi', 'kh', 'ng', 'nh', 'ph', 'qu', 'tr', 'th',
    'b', 'c', 'd', 'đ', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v', 'x'
]
vowels = (
    ['a', 'ă', 'â', 'e', 'ê', 'i', 'o', 'ô', 'ơ', 'u', 'ư', 'y'] +
    ['á', 'ắ', 'ấ', 'é', 'ế', 'í', 'ó', 'ố', 'ớ', 'ú', 'ứ', 'ý'] +
    ['à', 'ằ', 'ầ', 'è', 'ề', 'ì', 'ò', 'ồ', 'ờ', 'ù', 'ừ', 'ỳ'] +
    ['ả', 'ẳ', 'ẩ', 'ẻ', 'ể', 'ỉ', 'ỏ', 'ổ', 'ở', 'ủ', 'ử', 'ỷ'] +
    ['ã', 'ẵ', 'ẫ', 'ẽ', 'ễ', 'ĩ', 'õ', 'ỗ', 'ỡ', 'ũ', 'ữ', 'ỹ'] +
    ['ạ', 'ặ', 'ậ', 'ẹ', 'ệ', 'ị', 'ọ', 'ộ', 'ợ', 'ụ', 'ự', 'ỵ']
)

punctuations = ['.', '?', '"', '\'', ',', '-', '–', '!', ':', ';', '(', ')', '[', ']', '\n']

alphabet = sorted(set(''.join(consonants + vowels)))
# phonemes = sorted(consonants + vowels, key=len, reverse=True)
phonemes = consonants + vowels


def text_to_phonemes(text, keep_punctuation=False):
  text = unicodedata.normalize('NFKC', text.strip().lower())
  idx = 0
  out = []
  while idx < len(text):
    # length: 3, 2, 1
    for l in [3, 2, 1]:
      if idx + l <= len(text) and text[idx: (idx+l)] in phonemes:
        out.append(text[idx: (idx+l)])
        idx = idx + l
        break
    else:
      if idx < len(text):
        if keep_punctuation and text[idx] in punctuations:
          out.append(text[idx])
        if text[idx] == ' ':
          out.append(text[idx])
      idx = idx + 1
  return out


if __name__ == '__main__':
  text = """
Trăm năm trong cõi người ta,
Chữ tài chữ mệnh khéo là ghét nhau.
Trải qua một cuộc bể dâu,
Những điều trông thấy mà đau đớn lòng.
Lạ gì bỉ sắc tư phong,
Trời xanh quen thói má hồng đánh ghen.
Cảo thơm lần giở trước đèn,
Phong tình cổ lục còn truyền sử xanh.
Rằng: Năm Gia tĩnh triều Minh,
Bốn phương phẳng lặng hai kinh chữ vàng.
Có nhà viên ngoại họ Vương,
Gia tư nghỉ cũng thường thường bậc trung.
Một trai con thứ rốt lòng,
Vương Quan là chữ nối dòng nho gia.
Đầu lòng hai ả tố nga,
Thúy Kiều là chị em là Thúy Vân.
Mai cốt cách tuyết tinh thần,
Mỗi người một vẻ mười phân vẹn mười.
Vân xem trang trọng khác vời,
Khuôn trăng đầy đặn nét ngài nở nang.
Hoa cười ngọc thốt đoan trang,
Mây thua nước tóc tuyết nhường màu da.
Kiều càng sắc sảo mặn mà,
So bề tài sắc lại là phần hơn.
Làn thu thủy nét xuân sơn,
Hoa ghen thua thắm liễu hờn kém xanh.
"""
  print(text_to_phonemes(text, keep_punctuation=False))
