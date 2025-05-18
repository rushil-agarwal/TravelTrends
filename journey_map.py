def build_journey_stages(row):
    stages = ['Booking Confirmed']

    # if guest canceled the booking, their "journey" ends here
    if row['is_canceled'] == 1:
        stages.append('Canceled')
        return stages

    stages.append('Arrived')

    if row['amenity_utilization_ratio'] > 0:
        stages.append('Used Amenities')

    if row['simulated_complaints'] > 0:
        stages.append('Complained')

    stages.append('Check-Out')

    if row['is_repeated_guest'] == 1:
        stages.append('Repeat Guest')

    return stages