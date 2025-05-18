import pandas as pd

def clean_data(data):
    data['arrival_date'] = pd.to_datetime(data['arrival_date_year'].astype(str) + '-' + data['arrival_date_month'].astype(str) + '-' + data['arrival_date_day_of_month'].astype(str), errors = 'coerce')

    data['reservation_status_date'] = pd.to_datetime(data['reservation_status_date'].astype(str), errors='coerce')

    #used in future for calculating metrics and as features for predictions
    data['stay_length'] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']

    #replace null values with 0, and 1 in case of adults
    data['children'].fillna(0, inplace=True)
    data['babies'].fillna(0, inplace=True)
    data['adults'].fillna(1, inplace=True) 


    data['total_guests'] = data['adults'] + data['children'] + data['babies']
    data['is_family'] = ((data['children'] > 0) | (data['babies'] > 0)).astype(int)


    data['agent'].fillna(0, inplace=True)
    data['agent'] = data['agent'].astype(int).astype("category")

    data['company'].fillna(0, inplace=True)
    data['company'] = data['company'].astype(int).astype("category")


    data['country'].fillna("Unknown", inplace=True)
    data['country'] = data['country'].astype("category")


    data['season'] = data['arrival_date_month'].map({
        'December': 'Winter', 'January': 'Winter', 'February': 'Winter',
        'March': 'Spring', 'April': 'Spring', 'May': 'Spring',
        'June': 'Summer', 'July': 'Summer', 'August': 'Summer',
        'September': 'Fall', 'October': 'Fall', 'November': 'Fall'
    })

    toConvert = [ 'hotel', 'meal', 'market_segment', 'distribution_channel','reserved_room_type', 'assigned_room_type', 'deposit_type',
        'customer_type', 'reservation_status', 'season']
    
    for col in toConvert:
        data[col] = data[col].astype('category')

    return data
