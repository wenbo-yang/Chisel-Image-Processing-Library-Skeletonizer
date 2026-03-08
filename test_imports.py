#!/usr/bin/env python
import sys
sys.path.insert(0, '.')

print("Testing imports...")

try:
    from src.processor.blurer import ImageBlur
    print('✓ ImageBlur imported successfully')
except Exception as e:
    print(f'✗ ImageBlur error: {e}')

try:
    from src.processor.skeletonizer import Thinning
    print('✓ Thinning imported successfully')
except Exception as e:
    print(f'✗ Thinning error: {e}')

try:
    from src.processor.contour_grouper import ContourGrouper
    print('✓ ContourGrouper imported successfully')
except Exception as e:
    print(f'✗ ContourGrouper error: {e}')

try:
    from src.processor.edge_connector import EdgeConnector
    print('✓ EdgeConnector imported successfully')
except Exception as e:
    print(f'✗ EdgeConnector error: {e}')

try:
    from src.processor.edge_closer import EdgeCloser
    print('✓ EdgeCloser imported successfully')
except Exception as e:
    print(f'✗ EdgeCloser error: {e}')

print("\nAll tests completed.")
