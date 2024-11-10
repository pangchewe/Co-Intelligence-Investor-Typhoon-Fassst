import streamlit as st
import requests
import pandas as pd
import re
from typing import TypedDict
from string import Template
from openai import OpenAI

# Streamlit app starts here
st.title("SET Stock Analyzer (Typhoon)")


# User input for API keys
TYPHOON_API_KEY = "sk-JpuqAPZWPthc9KFfII8UZgaZdIRSNeZCyALfUNtKsAOhNL37"

# Credit Section
st.markdown("---")
st.markdown(
    """
    **Credits**  
    This application was developed by [Pang (QuantCorner)](https://www.linkedin.com/in/dhouch/), with special thanks to:
    - [**AJ.Pat** (InvestIdea)](https://web.facebook.com/investidea.in.th) for knowledge base 
    - [**P.Prem** (DataKarate)](https://web.facebook.com/datakarate/?_rdc=1&_rdr) for coding expertise 
    - [**P.Nut** (QuantCorner)](https://web.facebook.com/quantcornerthailand) for mentorship 
    """
)

# Initialize OpenAI client for Typhoon model
client = OpenAI(
    api_key=TYPHOON_API_KEY,
    base_url='https://api.opentyphoon.ai/v1'
)

class ChatTurn(TypedDict):
    role: str
    content: str


def get_open_ai_completion(
    prompt: str,
    model: str = "typhoon-v1.5x-70b-instruct",
    stream: bool = False,
    initial_message: list[ChatTurn] | None = None,
    temperature: float = 0
) -> str:
    if initial_message is None:
        initial_message = []
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, stream=stream
    )
    if not stream:
        return response.choices[0].message.content
    else:
        result = ""
        for chunk in response:
            content = chunk.choices[0].delta.content
            if isinstance(content, str):
                result += content
        return result

# User input for stock symbol
SYMBOL = st.text_input("Enter Stock Symbol (e.g., PTT):", "PTT").upper()

# Session setup
session = requests.Session()
session.get(f"https://www.settrade.com/th/equities/quote/{SYMBOL}/financial-statement/full")

headers = {
    'accept': 'application/json, text/plain, */*',
    'accept-language': 'en-TH,en;q=0.9',
    'referer': f'https://www.settrade.com/th/equities/quote/{SYMBOL}/financial-statement/full',
    'user-agent': 'Mozilla/5.0',
}

def get_sheet(period: str) -> pd.DataFrame:
    params = {
        'accountType': 'balance_sheet',
        'fsType': 'company',
        'period': period,
        'language': 'ENG',
    }

    response = session.get(
        f'https://www.settrade.com/api/set/stock/{SYMBOL}/financialstatement',
        params=params,
        headers=headers,
    )
    response.raise_for_status()
    data = response.json()
    return pd.DataFrame(data["accounts"])

def get_business_type(symbol: str) -> str:
    response = session.get(f"https://www.settrade.com/th/equities/quote/{symbol}/overview")
    response.raise_for_status()
    text = response.text
    pattern = r'businessType:\s*"([^"]+)"'
    found = re.search(pattern, text)

    if found:
        target_data = found.group(1)
        return target_data
    else:
        return "No business type found."

# Define the range of years
from_ = 2019
to = 2024
year_range = range(from_, to)
first_year, *year_range = year_range

# Fetch the balance sheet data
try:
    sheet = get_sheet(f"YE_{first_year}")[["accountCode", "accountName", "amount"]]
    sheet[str(first_year)] = sheet["amount"]
    sheet = sheet.drop("amount", axis=1)

    for year in year_range:
        sheet_by_year = get_sheet(f"YE_{year}")[["accountCode", "accountName", "amount"]]
        sheet_by_year[str(year)] = sheet_by_year["amount"]
        sheet_by_year = sheet_by_year.drop("amount", axis=1)

        sheet = pd.merge(sheet, sheet_by_year, on="accountCode", how='outer', suffixes=("", f"_{year}"))
        sheet['accountName'] = sheet['accountName'].fillna(sheet[f'accountName_{year}'])
        sheet = sheet.drop(f'accountName_{year}', axis=1)
    sheet = sheet.drop("accountCode", axis=1)
    sheet = sheet.set_index("accountName")
    sheet = sheet.replace({None: 0})
    sheet_prompt = sheet.reset_index().to_markdown(index=False, floatfmt=",.2f")
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()

# Fetch the business type
business_type = get_business_type(SYMBOL)

# Prepare the prompt
template = Template("""
# Context:

Stock name: ${SYMBOL}
Business type: ${business_type}

Balance sheet:
${sheet}

# Objective:
Answer the following questions:

${task}
""")

# Define all prompts (prompts 1 to 9)
prompt = template.safe_substitute(
    SYMBOL=SYMBOL,
    sheet=sheet_prompt,
    business_type=business_type,
    task="""
1.1 วิเคราะห์งบแสดงฐานะการเงิน
1.1.1 สินทรัพย์
1.1.1.1 ดูโครงสร้างของสินทรัพย์ ว่าสินทรัพย์หมุนเวียน หรือสินทรัพย์ไม่หมุนเวียนมากกว่ากัน ถ้าบริษัทมีสินทรัพย์หมุนเวียนมากให้เน้นวิเคราะห์สภาพคล่อง ถ้าบริษัทมีสินทรัพย์ไม่หมุนเวียนเยอะให้เน้นวิเคราะห์การนำสินทรัพย์ไปสร้างรายได้
1.1.1.2 วิเคราะห์สินทรัพย์หมุนเวียน รายการส่วนใหญ่ควรเป็น เงินสด ลูกหนี้การค้า และสินค้าคงเหลือ ถ้ามีรายการอื่นมากๆให้ระวังปัญหาสภาพคล่อง
1.1.1.3 บริษัทที่ลูกหนี้การค้าเพิ่มขึ้นมากกว่าการเพิ่มของยอดขาย และอัตราส่วนระยะเวลาเก็บหนี้เพิ่มขึ้น ให้ระวังปัญหาการเก็บหนี้ และการตกแต่งงบการเงินโดยการสร้างรายได้เทียม 
1.1.1.4 บริษัทที่สินค้าคงเหลือเพิ่มขึ้นมากกว่าการเพิ่มของต้นทุนขาย และอัตราส่วนระยะเวลาขายสินค้าเพิ่มขึ้น ให้ระวังปัญหาสินค้าล้าสมัยเสื่อมสภาพ
1.1.1.5 ให้ระวังบริษัทที่มีค่านิยมจำนวนมากเมื่อเทียบกับสินทรัพย์รวม เพราะอาจเป็นการเข้าซื้อกิจการราคาแพงได้ ให้ดูว่า ROA ยังสูงและสม่ำเสมอหรือไม่
1.1.1.6 บริษัทที่มีที่ดินอาคารอุปกรณ์เพิ่มสอดคล้องกับรายได้ที่เพิ่ม แสดงว่าบริษัทมีการลงทุนขยายกิจการให้เติบโต และรายได้เติบโตตาม 
"""
)

prompt2 = template.safe_substitute(
    SYMBOL=SYMBOL,
    sheet=sheet_prompt,
    business_type=business_type,
    task="""
1.1.2 วิเคราะห์หนี้สิน
1.1.2.1 ดูโครงสร้างหข้องหนี้สินว่าสอดคล้องกับโครงสร้างสินทรัพย์หรือไม่ 
1.1.2.1.1 ถ้าบริษัทที่มีสินทรัพย์หมุนเวียนมากกว่าสินทรัพย์ไม่หมุนเวียน หนี้สินหมุนเวียนจะมากกว่าหนี้สินไม่หมุนเวียน แสดงว่าโครงสร้างหนี้สินสอดคล้องกับสินทรัพย์
1.1.2.1.2 ถ้าบริษัทที่มีสินทรัพย์หมุนเวียนน้อยกว่าสินทรัพย์ไม่หมุนเวียน หนี้สินหมุนเวียนจะน้อยกว่าหนี้สินไม่หมุนเวียน แสดงว่าโครงสร้างหนี้สินสอดคล้องกับสินทรัพย์
1.1.2.1.3 ถ้าบริษัทมีสินทรัพย์ไม่หมุนเวียนเยอะ แต่หนี้ส่วนใหญ่เป็นหนี้ระยะสั้น ให้ระวังปัญหาหนี้สิน เนื่องจากใช้เงินผิดประเภทกู้ระยะสั้นมาลงทุนระยะยาว
1.1.2.2 ให้ระวังบริษัทที่มีส่วนของหนี้สินเพิ่มขึ้นเร็วกว่าส่วนของผู้ถือหุ้นที่เพิ่มขึ้น อาจมีปัญหาหนี้สินได้ในอนาคต
"""
)

prompt3 = template.safe_substitute(
    SYMBOL=SYMBOL,
    sheet=sheet_prompt,
    business_type=business_type,
    task="""
1.1.3 วิเคราะห์ส่วนของผู้ถือหุ้น
1.1.3.1 ให้ระวังบริษัทที่ขาดทุนสะสมจำนวนมากเมื่อเทียบกับทุนจดทะเบียน แสดงว่าในอดีตที่ผ่านมา มีผลประกอบการขาดทุนเป็นจำนวนมาก 
1.1.3.2 ให้ระวังบริษัทที่มี ตราสารหนี้เสมือนทุน จำนวนมากเมื่อเทียบกับส่วนของผู้ถือหุ้น เพราะจริงๆแล้วเป็นหนี้สินไม่ใช่ส่วนของผู้ถือหุ้น
"""
)

prompt4 = template.safe_substitute(
    SYMBOL=SYMBOL,
    sheet=sheet_prompt,
    business_type=business_type,
    task="""
1.2 วิเคราะห์งบกำไรขาดทุน
1.2.1 รายได้และกำไรสุทธิควรเพิ่มขึ้นอย่างต่อเนื่อง และอัตรากำไรขั้นต้น อัตรากำไรก่อนดอกเบี้ยและภาษี และอัตรากำไรสุทธิสม่ำเสมอ แสดงว่าเป็นบริษัทที่ดี เหมาะถือลงทุน
1.2.2 อัตรากำไรขั้นต้น ควรสม่ำเสมอหลายๆปีติดกัน
1.2.3 บริษัทที่อัตรากำไรขั้นต้นลดลงอาจมีปัญหาดังนี้
1.2.3.1 มีปัญหาต้นทุนหรือราคาสินค้าผันผวน
1.2.3.2 ยังใช้กำลังการผลิตไม่เต็มที่
1.2.3.3 มีปัญหาการแข่งขันกับคู่แข่งต้องลดแลกแจกแถมตัดราคาแข่งกัน
1.2.4 ค่าใช้จ่ายในการขายควรเพิ่มขึ้นสอดคล้องกับยอดขายที่เพิ่มขึ้น แสดงว่าใช้งบการตลาดได้คุ้มค่า
1.2.5 ค่าใช้จ่ายในการบริหารควรเพิ่มขึ้นสอดคล้องกับยอดขายที่เพิ่มขึ้น แสดงว่าบริหารงานส่วนกลางได้ดีสอดคล้องกับยอดขาย
1.2.6 ถ้าต้นทุนทางการเงินเกิน 50% ของกำไรก่อนดอกเบี้ยและภาษี แสดงว่ามีปัญหาจัดโครงสร้างหนี้สินมากเกินไป หรือกำลังอยู่ในช่วงแรกของการลงทุน 
"""
)

prompt5 = template.safe_substitute(
    SYMBOL=SYMBOL,
    sheet=sheet_prompt,
    business_type=business_type,
    task="""
1.3 วิเคราะห์งบกระแสเงินสด
1.3.1 งบกระแสเงินสดจากกิจกรรมควรเป็นบวก แสดงว่าบริษัททำธุรกิจแล้วมีเงินงอกเงยจากการดำเนินงาน สามารถนำเงินมาลงทุนและจ่ายหนี้ต่อได้
1.3.2 งบกระแสเงินสดจากกิจกรรมดำเนินงานที่น้อยกว่ากำไรสุทธิ แสดงว่าอาจมีกำไรพิเศษ เงินจมกับลูกหนี้การค้า สินค้าคงเหลือ หรือเป็นสัญญาณเบื้องต้นของการตกแต่งบัญชี
1.3.3 บริษัทที่มีเงินจ่ายกิจกรรมลงทุน รายการซื้อที่ดินอาคารอุปกรณ์ แปลงค่าเป็นบวก แล้วมากกว่าค่าเสื่อมราคา แสดงว่ามีการลงทุนสูง อาจอยู่ในช่วงเติบโตขยายกิจการได้
"""
)

prompt6 = template.safe_substitute(
    SYMBOL=SYMBOL,
    sheet=sheet_prompt,
    business_type=business_type,
    task="""
1.4 วิเคราะห์ปัญหาธุรกิจผ่านอัตราส่วนทางการเงิน
1.4.1 ผลตอบแทนต่อส่วนของผู้ถือหุ้น ROE > 5 และสม่ำเสมอ แสดงว่าบริษัทสามารถสร้างผลตอบแทนให้กับผู้ถือหุ้นได้ดี
1.4.2 ผลตอบแทนต่อสินทรัพย์ ROA > 5 และสม่ำเสมอ แสดงว่าบริษัทสามารถนำสินทรัพย์ไปสร้างผลตอบแทนกำไรก่อนดอกเบี้ยและภาษีได้ดี 
1.4.3 บริษัทที่ ROA ลดลงต่อเนื่องแสดงว่ามีปัญหาลงทุนแล้วไม่คุ้มค่าไม่สร้างผลตอบแทนกลับมา ผลประกอบการจะเริ่มโตช้าลงได้
1.4.4 บริษัทที่อัตราส่วนหนี้สินต่อทุนมากกว่า 1.5 และกระแสเงินสดจากกิจกรรมดำเนินงานติดลบ ให้ระมัดระวังปัญหาการจ่ายหนี้สิน
1.4.5 บริษัทที่อัตราส่วนหมุนเวียนทรัพย์สิน (asset turnover) มีแนวโน้มลดลง อาจอยู่ในช่วงลงทุน สินทรัพย์เพิ่มแต่รายได้ไม่เพิ่มตาม
1.4.6 ถ้าระยะเวลาเก็บหนี้เพิ่มให้ระวังปัญหาการเก็บหนี้
1.4.7 ถ้าระยะเวลาขายสินค้าเพิ่มขึ้นให้ระวังปัญหาสินค้าล้าสมัยเสื่อมสภาพ
1.4.8 ถ้าระยะเวลาชำระเจ้าหนี้เพิ่มขึ้นให้ระวังปัญหาสภาพคล่องไม่มีเงินจ่ายเจ้าหนี้การค้าต้องดึงรอบให้ยาวขึ้น
1.4.9 ถ้าบริษัทที่วงจรเงินสดยาวเกิน 75 วัน แสดงว่าต้องใช้เงินทุนหมุนเวียนเยอะ ให้ระวังปัญหาสภาพคล่อง
1.4.10 บริษัทที่อัตรากำไรขั้นต้นลดลงอาจมีปัญหา ต้นทุนวัตถุดิบราคาสินค้า หรือผลิตยังไม่เต็มกำลังการผลิต หรือปัญหาการแข่งขัน
1.4.11 ถ้าอัตรากำไรขั้นต้นใกล้เคียงเดิม แต่อัตรากำไรก่อนดอกเบี้ยและภาษีลดลง แสดงว่ามีปัญหาที่ค่าใช้จ่ายในการขายและบริหาร
1.4.12 ถ้าอัตรากำไรก่อนดอกเบี้ยและภาษีใกล้เคียงเดิม แต่อัตรากำไรสุทธิลดลง แสดงว่ามีปัญหาที่ดอกเบี้ยสูง จากโครงสร้างเงินทุนที่มีหนี้สินมาก
"""
)

prompt7 = template.safe_substitute(
    SYMBOL=SYMBOL,
    sheet=sheet_prompt,
    business_type=business_type,
    task="""
2. วิเคราะห์การเติบโต (Growth)

2.1 บริษัทอยู่ในอุตสาหกรรมที่กำลังเติบโต
2.2 บริษัทควรโตใกล้เคียงกับอุตสาหกรรม ถ้าบริษัทโตเร็วกว่าอุตสาหกรรม ไม่นานการเติบโตจะตันและค่อยๆ กลับมาอุตสาหกรรม ถ้าบริษัทเติบโตช้ากว่าอุตสาหกรรมไม่นานคู่แข่งจะเติบโตกว่าและชิงส่วนแบ่งการตลาดไป
2.3 บริษัทควรขยายสาขา ขยายโรงงานอย่างต่อเนื่องให้สอดคล้องกับอุตสาหกรรม
"""
)

prompt8 = template.safe_substitute(
    SYMBOL=SYMBOL,
    sheet=sheet_prompt,
    business_type=business_type,
    task="""
2.4 ลักษณะของงบการเงินที่เติบโตอย่างมีคุณภาพ
2.4.1 สินทรัพย์เพิ่มเรื่อยๆ แสดงว่ามีการลงทุนอย่างต่อเนื่อง
2.4.2 อัตราส่วนหมุนเวียนสินทรัพย์คงที่ แสดงว่าสินทรัพย์เพิ่มและรายได้เพิ่มไปในทิศทางเดียวกัน
2.4.3 อัตรากำไรสุทธิคงที่ แสดงว่ามีการควบคุมค่าใช้จ่ายภายในได้ดี
2.4.4 อัตราส่วนหนี้สินต่อทุนสม่ำเสมอแสดงว่าจัดวางโครงสร้างทางการเงินได้ดี
2.5 ลักษณะงบการเงินของหุ้นที่กำลังเติบโตลดลง
2.5.1 ROA เริ่มลดลงต่อเนื่องหลายปี
2.5.2 อัตราส่วนหมุนเวียนสินทรัพย์ลดลง จากสินทรัพย์เพิ่มแล้วรายได้ไม่เพิ่มตาม
2.5.3 อัตรากำไรลดลง จากการมีรายจ่ายเพื่อการเติบโตเพิ่มแต่รายได้ไม่โตตาม
"""
)

prompt9 = template.safe_substitute(
    SYMBOL=SYMBOL,
    sheet=sheet_prompt,
    business_type=business_type,
    task="""
3. ประเมินมูลค่าหุ้นความถูกแพง (Value)

3.1 ความถูกแพงวัดจาก PE Ratio
3.2 หุ้นที่ PE Ratio สูงเกิน 20 แสดงว่าตลาดให้ความสนใจกับการเติบโตในอนาคตมากกว่ากำไรในปัจจุบัน
3.3 หุ้นที่ PE Ratio ต่ำกว่า 20 แสดงว่าตลาดให้ความสนใจกับกำไรในปัจจุบัน ให้ระวัง
3.3.1 หุ้น PE Ratio ต่ำเพราะกำไรที่มาครั้งเดียว เช่นกำไรจากการขายสินทรัพย์ กำไรจากอัตราแลกเปลี่ยน 
3.3.2 หุ้น PE Ratio ต่ำเพราะกำไรจากราคาสินค้าพุ่ง อาจมาแค่ครั้งเดียวแล้วก็หาย
3.3.3 หุ้น PE Ratio ต่ำเพราะกำไรผันผวนไม่สม่ำเสมอ ตลาดต้องให้ PE ต่ำเพื่อชดเชยความเสี่ยงที่กำไรผันผวน
"""
)

# Add all prompts to the analysis options
analysis_options = {
    "วิเคราะห์ฐานะการเงิน": prompt,
    "วิเคราะห์หนี้สิน": prompt2,
    "วิเคราะห์ส่วนของผู้ถือหุ้น": prompt3,
    "วิเคราะห์งบกำไรขาดทุน": prompt4,
    "วิเคราะห์งบกระแสเงินสด": prompt5,
    "วิเคราะห์อัตราส่วนทางการเงิน": prompt6,
    "วิเคราะห์การเติบโต": prompt7,
    "วิเคราะห์งบการเงินที่เติบโต": prompt8,
    "ประเมินมูลค่าหุ้นความถูกแพง": prompt9,
}

# Let the user select an analysis type
selected_analysis = st.selectbox(
    "Select the type of analysis you want to perform:",
    list(analysis_options.keys())
)

# Generate the response only for the selected analysis
if st.button("Generate Analysis"):
    selected_prompt = analysis_options[selected_analysis]
    try:
        with st.spinner('Generating analysis...'):
            output = get_open_ai_completion(selected_prompt)
        # Display the selected analysis response
        st.subheader(f"{selected_analysis} Response:")
        st.markdown(output)
    except Exception as e:
        st.error(f"An error occurred while generating the analysis: {e}")

# Display the business type
st.subheader(f"Business Type for {SYMBOL}:")
st.write(business_type)

# Display the balance sheet
st.subheader(f"Balance Sheet for {SYMBOL}:")
st.dataframe(sheet.style.format("{:,.2f}"))
